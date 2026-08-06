import argparse
import asyncio
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmark,
    ImageSegmenter,
    ImageSegmenterOptions,
    RunningMode,
)
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pyngrok import ngrok

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("tvm-server")

# --- Product Catalog ---
class Product(BaseModel):
    id: str
    name: str
    brand: str
    price: int
    image_url: str
    color_primary: str
    color_secondary: str

PRODUCTS: List[Product] = [
    Product(
        id="tuxedo",
        name="Black Tuxedo & Bow Tie",
        brand="Decart Edition",
        price=0,
        image_url="/static/products/tuxedo.jpg",
        color_primary="#000000",
        color_secondary="#FFFFFF"
    ),
    Product(
        id="shirt_mustard_polo",
        name="Mustard Prime Club Polo",
        brand="Gul Ahmed Men Edition",
        price=3800,
        image_url="/static/products/shirt_mustard_polo.jpg",
        color_primary="#8B8000",
        color_secondary="#FFFFFF"
    ),
    Product(
        id="shirt_black_polo",
        name="Minimalist Black Polo",
        brand="Khaadi Men Collection",
        price=3200,
        image_url="/static/products/shirt_black_polo.jpg",
        color_primary="#121212",
        color_secondary="#CCCCCC"
    ),
    Product(
        id="suit_teal_coral",
        name="Teal & Coral Botanical",
        brand="Khaadi Unstitched Lawn 2025",
        price=4500,
        image_url="/static/products/suit_teal_coral.jpg",
        color_primary="#008080",
        color_secondary="#FF7F50"
    ),
    Product(
        id="suit_emerald_gold",
        name="Emerald & Gold Heritage",
        brand="Gul Ahmed Premium Lawn",
        price=5500,
        image_url="/static/products/suit_emerald_gold.jpg",
        color_primary="#50C878",
        color_secondary="#FFD700"
    ),
    Product(
        id="suit_maroon_ivory",
        name="Maroon & Ivory Royale",
        brand="Khaadi Pret Collection",
        price=7500,
        image_url="/static/products/suit_maroon_ivory.jpg",
        color_primary="#800000",
        color_secondary="#FFFFF0"
    )
]

PRODUCTS_DICT = {p.id: p for p in PRODUCTS}

# --- Directory Setup & Synthetic Garment Generation ---
STATIC_DIR = Path("static")
PRODUCTS_DIR = STATIC_DIR / "products"
GARMENTS_DIR = PRODUCTS_DIR / "garments"

def ensure_directories():
    STATIC_DIR.mkdir(exist_ok=True)
    PRODUCTS_DIR.mkdir(exist_ok=True)
    GARMENTS_DIR.mkdir(exist_ok=True)

def hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])  # BGR for OpenCV

def generate_synthetic_garments():
    ensure_directories()
    for product in PRODUCTS:
        garment_path = GARMENTS_DIR / f"{product.id}.png"
        prod_img_path = STATIC_DIR / product.image_url.lstrip("/")

        if prod_img_path.exists():
            base_img = cv2.imread(str(prod_img_path))
            if base_img is not None:
                gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
                bg_mask = (gray > 232).astype(np.uint8) * 255
                fg_mask = cv2.bitwise_not(bg_mask)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

                coords = cv2.findNonZero(fg_mask)
                if coords is not None:
                    x, y, bw, bh = cv2.boundingRect(coords)
                    shirt_crop = base_img[y:y+bh, x:x+bw]
                    alpha_crop = fg_mask[y:y+bh, x:x+bw]

                    shirt_resized = cv2.resize(shirt_crop, (512, 640), interpolation=cv2.INTER_AREA)
                    alpha_resized = cv2.resize(alpha_crop, (512, 640), interpolation=cv2.INTER_AREA)
                    alpha_resized = cv2.GaussianBlur(alpha_resized, (5, 5), 0)

                    img = np.zeros((640, 512, 4), dtype=np.uint8)
                    img[:, :, :3] = shirt_resized
                    img[:, :, 3] = alpha_resized

                    cv2.imwrite(str(garment_path), img)
                    logger.info(f"Generated clean background-removed garment: {product.id}")
                    continue

        # Fallback if product photo missing
        width, height = 512, 640
        img = np.zeros((height, width, 4), dtype=np.uint8)
        c1 = hex_to_bgr(product.color_primary)
        c2 = hex_to_bgr(product.color_secondary)
        for y in range(height):
            ratio = y / height
            b = min(255, int(c1[0] * (1 - ratio) + c2[0] * ratio) + 30)
            g = min(255, int(c1[1] * (1 - ratio) + c2[1] * ratio) + 30)
            r = min(255, int(c1[2] * (1 - ratio) + c2[2] * ratio) + 30)
            img[y, :] = [b, g, r, 255]
        cv2.imwrite(str(garment_path), img)

    logger.info(f"All garment templates ready in {GARMENTS_DIR}")

# --- Model Path ---
MODEL_DIR = Path("models")
POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker_lite.task"

def ensure_pose_model():
    """Download pose landmarker model if not present."""
    MODEL_DIR.mkdir(exist_ok=True)
    if not POSE_MODEL_PATH.exists():
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        logger.info(f"Downloading pose model from {url}...")
        urllib.request.urlretrieve(url, str(POSE_MODEL_PATH))
        logger.info("Pose model downloaded.")

ENVIRONMENTS_DIR = STATIC_DIR / "environments"

class CustomGarmentUploadRequest(BaseModel):
    image: str  # base64 image of clothing
    name: str

# Cache environment images
ENVIRONMENTS_CACHE: Dict[str, np.ndarray] = {}

def load_environments():
    ENVIRONMENTS_DIR.mkdir(exist_ok=True)
    for env_name in ["khaadi_store", "fashion_runway", "royal_palace", "studio"]:
        path = ENVIRONMENTS_DIR / f"{env_name}.jpg"
        if path.exists():
            img = cv2.imread(str(path))
            if img is not None:
                ENVIRONMENTS_CACHE[env_name] = img

load_environments()

# Optional PyTorch CUDA LCM Diffusion Pipeline for Colab GPU execution
HAS_TORCH_CUDA = False
torch_lcm_pipe = None

def init_lcm_cuda():
    global HAS_TORCH_CUDA, torch_lcm_pipe
    try:
        import torch
        from diffusers import AutoPipelineForImage2Image, LCMScheduler
        if torch.cuda.is_available():
            logger.info("⚡ CUDA GPU detected! Initializing PyTorch LCM Video Diffusion Engine...")
            model_id = "Lykon/dreamshaper-8"
            pipe = AutoPipelineForImage2Image.from_pretrained(
                model_id, torch_dtype=torch.float16, safety_checker=None
            ).to("cuda")
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
            torch_lcm_pipe = pipe
            HAS_TORCH_CUDA = True
            logger.info("⚡ Real-Time PyTorch LCM Video-to-Video Engine READY on GPU!")
    except Exception as e:
        logger.info(f"PyTorch CUDA LCM Pipeline fallback to CPU mode: {e}")

# --- Try-On Engine (Lucy Decart AI Architecture) ---
class TryOnEngine:
    def __init__(self):
        init_lcm_cuda()
        ensure_pose_model()
        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=True,  # Enabled for World Background Editing
        )
        self.pose_landmarker = PoseLandmarker.create_from_options(pose_options)
        self.garments: Dict[str, np.ndarray] = {}
        self.load_garments()
        self._log_count = 0
        logger.info("TryOnEngine initialized with World Editing & Segmentation capabilities.")

    def load_garments(self):
        for product in PRODUCTS:
            path = str(GARMENTS_DIR / f"{product.id}.png")
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    self.garments[product.id] = img
                    logger.info(f"Loaded garment: {product.id} shape={img.shape}")

    def add_custom_garment(self, garment_id: str, img_b64: str):
        try:
            if "," in img_b64:
                img_b64 = img_b64.split(",")[1]
            raw_bytes = base64.b64decode(img_b64)
            np_arr = np.frombuffer(raw_bytes, np.uint8)
            base_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if base_img is not None:
                width, height = 512, 640
                base_resized = cv2.resize(base_img, (width, height), interpolation=cv2.INTER_AREA)

                img = np.zeros((height, width, 4), dtype=np.uint8)
                img[:, :, :3] = base_resized

                shirt_mask = np.zeros((height, width), dtype=np.uint8)
                points = np.array([
                    [width * 0.12, height * 0.0],
                    [width * 0.38, height * 0.0],
                    [width * 0.50, height * 0.08],
                    [width * 0.62, height * 0.0],
                    [width * 0.88, height * 0.0],
                    [width * 0.98, height * 0.12],
                    [width * 0.92, height * 0.35],
                    [width * 0.78, height * 0.28],
                    [width * 0.82, height * 1.0],
                    [width * 0.18, height * 1.0],
                    [width * 0.22, height * 0.28],
                    [width * 0.08, height * 0.35],
                    [width * 0.02, height * 0.12],
                ], np.int32)
                cv2.fillPoly(shirt_mask, [points], 255)
                shirt_mask = cv2.GaussianBlur(shirt_mask, (7, 7), 0)
                img[:, :, 3] = shirt_mask

                garment_path = GARMENTS_DIR / f"{garment_id}.png"
                cv2.imwrite(str(garment_path), img)
                self.garments[garment_id] = img
                logger.info(f"Custom garment {garment_id} added successfully.")
                return True
        except Exception as e:
            logger.error(f"Error adding custom garment: {e}")
        return False

    def process_frame(self, frame_b64: str, product_id: str, prompt: str = "", environment: str = "none") -> tuple[Optional[str], float]:
        start_time = time.time()
        self._log_count += 1

        # Check for Real-Time Generative AI Diffusion Stream (Fal.ai Key)
        fal_key = os.environ.get("FAL_KEY", "")
        if fal_key:
            try:
                import fal_client
                prompt_text = prompt or f"photorealistic 2.5d video of person wearing {product_id} tuxedo suit outfit, natural fabric draping, realistic clothing folds"
                res = fal_client.subscribe(
                    "fal-ai/fast-sdxl/image-to-image",
                    arguments={
                        "image_url": f"data:image/jpeg;base64,{frame_b64}",
                        "prompt": prompt_text,
                        "strength": 0.50,
                        "num_inference_steps": 4,
                        "guidance_scale": 1.5
                    }
                )
                if "images" in res and len(res["images"]) > 0:
                    gen_url = res["images"][0]["url"]
                    import urllib.request
                    with urllib.request.urlopen(gen_url) as resp:
                        gen_bytes = resp.read()
                        out_b64 = base64.b64encode(gen_bytes).decode("utf-8")
                        latency_ms = (time.time() - start_time) * 1000
                        return out_b64, latency_ms
            except Exception as e:
                logger.error(f"Generative Fal.ai streaming error: {e}")

        # Check for PyTorch CUDA LCM Generative Stream (Colab GPU Execution)
        if HAS_TORCH_CUDA and torch_lcm_pipe is not None:
            try:
                from PIL import Image
                if "," in frame_b64:
                    frame_b64 = frame_b64.split(",")[1]
                img_bytes = base64.b64decode(frame_b64)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame_cv is not None:
                    pil_img = Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))
                    prompt_text = prompt or f"photorealistic 2.5d video frame of person wearing luxurious {product_id} suit outfit, natural fabric draping, realistic clothing folds"
                    gen_result = torch_lcm_pipe(
                        prompt=prompt_text,
                        image=pil_img,
                        num_inference_steps=4,
                        guidance_scale=1.2,
                        strength=0.45
                    ).images[0]
                    out_bgr = cv2.cvtColor(np.array(gen_result), cv2.COLOR_RGB2BGR)
                    _, buffer = cv2.imencode('.jpg', out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    out_b64 = base64.b64encode(buffer).decode('utf-8')
                    latency_ms = (time.time() - start_time) * 1000
                    return out_b64, latency_ms
            except Exception as e:
                logger.error(f"PyTorch CUDA LCM Streaming error: {e}")

        try:
            if "," in frame_b64:
                frame_b64 = frame_b64.split(",")[1]
            img_data = base64.b64decode(frame_b64)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                raise ValueError("Could not decode image")

            h, w = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            results = self.pose_landmarker.detect(mp_image)

            # --- 1. World Environment Background Swap ---
            if environment in ENVIRONMENTS_CACHE and results.segmentation_masks and len(results.segmentation_masks) > 0:
                bg_img = ENVIRONMENTS_CACHE[environment]
                bg_resized = cv2.resize(bg_img, (w, h))

                seg_mask_mp = results.segmentation_masks[0]
                seg_mask = seg_mask_mp.numpy_view().copy()
                if seg_mask.shape[:2] != (h, w):
                    seg_mask = cv2.resize(seg_mask, (w, h))

                seg_mask = cv2.GaussianBlur(seg_mask, (11, 11), 0)
                seg_mask_3c = np.dstack([seg_mask] * 3)

                # Composite person on top of environment background
                frame = (frame.astype(np.float32) * seg_mask_3c + bg_resized.astype(np.float32) * (1.0 - seg_mask_3c)).astype(np.uint8)

            # Map mode shortcuts
            if product_id == "khaadi":
                product_id = "suit_teal_coral"

            # --- 1. Animated / Pixar 3D Movie Transformation Shader ---
            if product_id == "animated" or "animated" in prompt.lower() or "pixar" in prompt.lower():
                color = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.medianBlur(gray, 7)
                edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
                edges_3c = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                cartoon = cv2.bitwise_and(color, edges_3c)
                # Warm 3D Pixar Studio lighting boost
                frame = cv2.addWeighted(cartoon, 0.82, np.full_like(cartoon, (30, 80, 140)), 0.18, 0)

            # --- 2. Baby Alien Shader ---
            elif product_id == "baby_alien" or "alien" in prompt.lower():
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + 35) % 180  # Alien Green skin tint
                alien = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                frame = cv2.addWeighted(frame, 0.3, alien, 0.7, 0)

            # --- 3. Green Slime Shader ---
            elif product_id == "green_slime" or "slime" in prompt.lower():
                slime = np.zeros_like(frame)
                slime[:, :, 1] = 230  # Neon Green
                slime[:, :, 0] = 50
                frame = cv2.addWeighted(frame, 0.65, slime, 0.35, 0)

            # --- 4. Hair on Fire Shader ---
            elif product_id == "hair_on_fire" or "fire" in prompt.lower():
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + 160) % 180  # Flame red tint
                fire = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                frame = cv2.addWeighted(frame, 0.5, fire, 0.5, 0)

            has_pose = results.pose_landmarks and len(results.pose_landmarks) > 0
            has_garment = product_id in self.garments

            # --- 2. Outfit Warping & Prompt Guided Texture Modification ---
            if has_pose and has_garment:
                landmarks = results.pose_landmarks[0]

                l_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
                r_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
                l_hip = landmarks[PoseLandmark.LEFT_HIP]
                r_hip = landmarks[PoseLandmark.RIGHT_HIP]

                shoulders_visible = (l_shoulder.visibility > 0.3 and r_shoulder.visibility > 0.3)
                hips_visible = (l_hip.visibility > 0.3 and r_hip.visibility > 0.3)

                if shoulders_visible:
                    shoulder_width = abs(l_shoulder.x - r_shoulder.x)
                    mid_x = (l_shoulder.x + r_shoulder.x) / 2

                    # Check face/chin landmarks for close-up framing positioning
                    nose = landmarks[PoseLandmark.NOSE]
                    m_left = landmarks[PoseLandmark.MOUTH_LEFT]
                    m_right = landmarks[PoseLandmark.MOUTH_RIGHT]

                    has_face = (nose.visibility > 0.3 and m_left.visibility > 0.3)
                    if has_face:
                        chin_y = (m_left.y + m_right.y) / 2
                        # Collarbone / neckline starts right below chin
                        neck_top_y = chin_y + 0.06
                    else:
                        neck_top_y = (l_shoulder.y + r_shoulder.y) / 2 - 0.08

                    # Ensure top of suit sits at collarbone level
                    top_y = min(neck_top_y, (l_shoulder.y + r_shoulder.y) / 2 - 0.05)

                    if hips_visible:
                        lh_x, lh_y = l_hip.x, l_hip.y
                        rh_x, rh_y = r_hip.x, r_hip.y
                    else:
                        # For close-up webcam frames, extend suit down to bottom of viewport
                        hip_offset_y = max(shoulder_width * 1.5, 0.95 - top_y)
                        hip_half_w = shoulder_width * 0.52
                        lh_x = mid_x + hip_half_w
                        lh_y = top_y + hip_offset_y
                        rh_x = mid_x - hip_half_w
                        rh_y = top_y + hip_offset_y

                    margin_x = 0.30 * shoulder_width

                    dst_points = np.array([
                        [(r_shoulder.x - margin_x) * w, top_y * h],
                        [(l_shoulder.x + margin_x) * w, top_y * h],
                        [(lh_x + margin_x) * w, lh_y * h],
                        [(rh_x - margin_x) * w, rh_y * h],
                    ], dtype=np.float32)

                    garment = self.garments.get(product_id)
                    gar_path = GARMENTS_DIR / f"{product_id}.png"
                    if gar_path.exists():
                        fresh_g = cv2.imread(str(gar_path), cv2.IMREAD_UNCHANGED)
                        if fresh_g is not None:
                            garment = fresh_g
                            self.garments[product_id] = fresh_g

                    if garment is None:
                        garment = np.zeros((640, 512, 4), dtype=np.uint8)
                    else:
                        garment = garment.copy()

                    # Dynamic Prompt-Guided Color / Tint Modification
                    if prompt:
                        prompt_lower = prompt.lower()
                        tint = None
                        if "red" in prompt_lower or "maroon" in prompt_lower or "ruby" in prompt_lower:
                            tint = (30, 30, 200)
                        elif "green" in prompt_lower or "emerald" in prompt_lower:
                            tint = (50, 180, 50)
                        elif "gold" in prompt_lower or "yellow" in prompt_lower or "amber" in prompt_lower:
                            tint = (30, 215, 255)
                        elif "blue" in prompt_lower or "teal" in prompt_lower or "cyan" in prompt_lower:
                            tint = (200, 180, 40)
                        elif "purple" in prompt_lower or "velvet" in prompt_lower:
                            tint = (180, 40, 140)

                        if tint is not None:
                            rgb_layer = garment[:, :, :3].astype(np.float32)
                            tint_arr = np.full_like(rgb_layer, tint, dtype=np.float32)
                            garment[:, :, :3] = np.clip(cv2.addWeighted(rgb_layer, 0.6, tint_arr, 0.4, 0), 0, 255).astype(np.uint8)

                    gh, gw = garment.shape[:2]
                    src_points = np.array([
                        [0,      0],
                        [gw - 1, 0],
                        [gw - 1, gh - 1],
                        [0,      gh - 1]
                    ], dtype=np.float32)

                    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                    warped = cv2.warpPerspective(garment, matrix, (w, h),
                                                flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT,
                                                borderValue=(0, 0, 0, 0))

                    if warped.shape[2] == 4:
                        alpha = warped[:, :, 3].astype(np.float32) / 255.0
                        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
                        alpha = np.clip(alpha * 1.15, 0, 1.0)
                        alpha_3c = np.dstack([alpha] * 3)

                        warped_rgb = warped[:, :, :3].astype(np.float32)
                        frame_f = frame.astype(np.float32)

                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                        gray_blur = cv2.GaussianBlur(gray_frame, (21, 21), 0)
                        shading_map = np.clip(gray_blur * 0.6 + 0.5, 0.65, 1.25)
                        shading_3c = np.dstack([shading_map] * 3)

                        warped_shaded = np.clip(warped_rgb * shading_3c, 0, 255)
                        blended = warped_shaded * alpha_3c + frame_f * (1.0 - alpha_3c)
                        frame = blended.astype(np.uint8)

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            out_b64 = base64.b64encode(buffer).decode('utf-8')

            latency_ms = (time.time() - start_time) * 1000
            return out_b64, latency_ms

        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            return None, 0.0

# --- FastAPI App ---
app = FastAPI(title="TVM Live Try-On")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Engine initialized lazily or on startup
engine: TryOnEngine = None

@app.on_event("startup")
async def startup_event():
    global engine
    generate_synthetic_garments()
    engine = TryOnEngine()
    logger.info("TryOnEngine initialized.")

class FalKeyRequest(BaseModel):
    key: str

@app.post("/api/set-fal-key")
async def set_fal_key(req: FalKeyRequest):
    os.environ["FAL_KEY"] = req.key.strip()
    logger.info("Fal.ai Real-Time Generative Stream Key activated!")
    return {"status": "success", "message": "Generative AI Video-to-Video Engine Activated!"}

class HDTryOnRequest(BaseModel):
    image: str
    product_id: str

def _run_hd_tryon_sync(frame_b64: str, product_id: str) -> Optional[str]:
    try:
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",")[1]

        user_img_bytes = base64.b64decode(frame_b64)
        temp_user_path = str(STATIC_DIR / "temp_user_hd.jpg")
        with open(temp_user_path, "wb") as f:
            f.write(user_img_bytes)

        product = PRODUCTS_DICT.get(product_id)
        if not product:
            return None

        garm_path = str(STATIC_DIR / product.image_url.lstrip("/"))
        if not os.path.exists(garm_path):
            garm_path = str(GARMENTS_DIR / f"{product_id}.png")

        logger.info(f"Triggering IDM-VTON Neural Fit for product {product_id}...")
        from gradio_client import Client, handle_file

        client = Client("yisol/IDM-VTON")
        res = client.predict(
            dict={"background": handle_file(temp_user_path), "layers": [], "composite": None},
            garm_img=handle_file(garm_path),
            garment_des=f"{product.brand} {product.name}",
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=25,
            seed=42,
            api_name="/tryon"
        )

        out_file = res[0]
        logger.info(f"IDM-VTON Neural Fit finished: {out_file}")

        with open(out_file, "rb") as f:
            res_bytes = f.read()

        return base64.b64encode(res_bytes).decode("utf-8")
    except Exception as e:
        logger.error(f"IDM-VTON endpoint error: {e}", exc_info=True)
        # Fallback to high-quality 2.5D shading synthesis if remote GPU space is busy
        if engine:
            out_b64, _ = engine.process_frame(frame_b64, product_id)
            return out_b64
        return None

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/api/products", response_model=List[Product])
async def get_products():
    return PRODUCTS

@app.post("/api/hd-tryon")
async def hd_tryon(req: HDTryOnRequest):
    loop = asyncio.get_running_loop()
    result_b64 = await loop.run_in_executor(None, _run_hd_tryon_sync, req.image, req.product_id)
    if result_b64:
        return {"status": "success", "image": result_b64}
    return {"status": "error", "message": "Neural try-on processing failed."}

@app.post("/api/upload-garment")
async def upload_garment(req: CustomGarmentUploadRequest):
    garment_id = f"custom_{int(time.time())}"
    success = engine.add_custom_garment(garment_id, req.image)
    if success:
        new_prod = Product(
            id=garment_id,
            name=req.name or "Custom Outfit",
            brand="Custom Upload",
            price=0,
            image_url=f"/static/products/garments/{garment_id}.png",
            color_primary="#c59b27",
            color_secondary="#ffffff"
        )
        PRODUCTS.append(new_prod)
        PRODUCTS_DICT[garment_id] = new_prod
        return {"status": "success", "product": new_prod.dict()}
    return {"status": "error", "message": "Failed to process custom outfit image"}

@app.websocket("/ws/tryon")
async def websocket_tryon(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected.")

    current_product_id = PRODUCTS[0].id
    current_prompt = ""
    current_environment = "none"

    frame_queue = asyncio.Queue(maxsize=2)

    # Task to process frames
    async def process_task():
        last_frame_time = time.time()
        frames_processed = 0
        fps = 0.0

        while True:
            try:
                frame_b64, prod_id, prompt_text, env_name = await frame_queue.get()

                loop = asyncio.get_running_loop()
                out_b64, latency_ms = await loop.run_in_executor(
                    None, engine.process_frame, frame_b64, prod_id, prompt_text, env_name
                )

                frames_processed += 1
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed > 1.0:
                    fps = frames_processed / elapsed
                    frames_processed = 0
                    last_frame_time = current_time

                if out_b64:
                    await websocket.send_json({
                        "type": "frame",
                        "data": out_b64,
                        "fps": round(fps, 1),
                        "latency_ms": round(latency_ms, 1)
                    })

                frame_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Process task error: {e}")

    processor = asyncio.create_task(process_task())

    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
                msg_type = data.get("type")

                if msg_type == "select_product":
                    pid = data.get("product_id")
                    if pid in PRODUCTS_DICT:
                        current_product_id = pid
                        logger.info(f"Product switched to {current_product_id}")
                elif msg_type == "frame":
                    frame_data = data.get("data")
                    pid = data.get("product_id", current_product_id)
                    current_prompt = data.get("prompt", current_prompt)
                    current_environment = data.get("environment", current_environment)

                    if pid in PRODUCTS_DICT:
                        current_product_id = pid

                    if frame_data:
                        try:
                            frame_queue.put_nowait((frame_data, current_product_id, current_prompt, current_environment))
                        except asyncio.QueueFull:
                            try:
                                frame_queue.get_nowait()
                                frame_queue.task_done()
                                frame_queue.put_nowait((frame_data, current_product_id, current_prompt, current_environment))
                            except (asyncio.QueueEmpty, asyncio.QueueFull):
                                pass
            except json.JSONDecodeError:
                logger.warning("Invalid JSON received")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected gracefully.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        processor.cancel()

# Mount static files at the end to not override routes
app.mount("/static", StaticFiles(directory="static"), name="static")


def main():
    parser = argparse.ArgumentParser(description="TVM Live Try-On Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    parser.add_argument("--ngrok-token", type=str, help="Ngrok auth token")
    parser.add_argument("--no-tunnel", action="store_true", help="Disable Ngrok tunnel")
    args = parser.parse_args()

    # Ensure static directories exist before mounting
    ensure_directories()

    local_url = f"http://localhost:{args.port}"
    public_url = local_url

    if not args.no_tunnel:
        if args.ngrok_token:
            ngrok.set_auth_token(args.ngrok_token)
        try:
            tunnel = ngrok.connect(args.port)
            public_url = tunnel.public_url
        except Exception as e:
            logger.error(f"Failed to start Ngrok tunnel: {e}")

    logger.info("="*50)
    logger.info(f"🚀 Server starting up...")
    logger.info(f"📍 Local URL:  {local_url}")
    if not args.no_tunnel and public_url != local_url:
        logger.info(f"🌍 Public URL: {public_url}")
    logger.info("="*50)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")

if __name__ == "__main__":
    main()
