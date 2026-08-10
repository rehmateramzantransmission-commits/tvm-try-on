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

# --- TurboJPEG Fast Frame Codec ---
HAS_TURBOJPEG = False
jpeg_codec = None

try:
    from turbojpeg import TurboJPEG
    jpeg_codec = TurboJPEG()
    HAS_TURBOJPEG = True
    logger.info("⚡ TurboJPEG fast codec initialized!")
except Exception as e:
    logger.info(f"TurboJPEG native C-library fallback to OpenCV JPEG: {e}")

def decode_jpeg(jpeg_bytes: bytes) -> Optional[np.ndarray]:
    if HAS_TURBOJPEG and jpeg_codec is not None:
        try:
            return jpeg_codec.decode(jpeg_bytes)
        except Exception:
            pass
    np_arr = np.frombuffer(jpeg_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def encode_jpeg(img: np.ndarray, quality: int = 80) -> bytes:
    if HAS_TURBOJPEG and jpeg_codec is not None:
        try:
            return jpeg_codec.encode(img, quality=quality)
        except Exception:
            pass
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes()

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

                    shirt_resized = cv2.resize(shirt_crop, (512, 512), interpolation=cv2.INTER_AREA)
                    alpha_resized = cv2.resize(alpha_crop, (512, 512), interpolation=cv2.INTER_AREA)
                    alpha_resized = cv2.GaussianBlur(alpha_resized, (5, 5), 0)

                    img = np.zeros((512, 512, 4), dtype=np.uint8)
                    img[:, :, :3] = shirt_resized
                    img[:, :, 3] = alpha_resized

                    cv2.imwrite(str(garment_path), img)
                    logger.info(f"Generated clean 512x512 garment cutout: {product.id}")
                    continue

        # Fallback if product photo missing
        width, height = 512, 512
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

# Import the refactored engine (face-safe, pose-guided, inpaint-ready)
from tryon_engine import TryOnEngine as _TryOnEngineImpl

# --- Try-On Engine wrapper (delegates to tryon_engine.TryOnEngine) ---
class TryOnEngine:
    def __init__(self):
        self._engine = _TryOnEngineImpl(
            garments_dir=GARMENTS_DIR,
            products=PRODUCTS,
        )
        self.garments = self._engine.garments
        self._log_count = 0
        logger.info("TryOnEngine initialized.")

    def load_garments(self):
        """Reload garments from disk into the inner engine."""
        self._engine._load_garments(GARMENTS_DIR, PRODUCTS)
        self.garments = self._engine.garments

    def add_custom_garment(self, garment_id: str, img_b64: str):
        ok = self._engine.add_custom_garment(garment_id, img_b64, GARMENTS_DIR)
        if ok:
            self.garments = self._engine.garments
        return ok

    def process_frame(
        self,
        frame_b64: str,
        product_id: str,
        prompt: str = "",
        environment: str = "none",
    ) -> tuple[Optional[str], float]:
        """
        Decode base64 frame → run TryOnEngine → return base64 output + latency.
        Face pixels are ALWAYS composited back from the raw camera input.
        """
        start_time = time.time()
        self._log_count += 1

        try:
            # --- Decode input frame -----------------------------------------
            raw_b64 = frame_b64.split(",")[1] if "," in frame_b64 else frame_b64
            img_data = base64.b64decode(raw_b64)
            frame_bgr = decode_jpeg(img_data)

            if frame_bgr is None:
                raise ValueError("Could not decode incoming frame")

            # Standardize to 512×512 for fast processing
            frame_bgr = cv2.resize(frame_bgr, (512, 512), interpolation=cv2.INTER_AREA)

            # Map legacy mode shortcuts
            if product_id == "khaadi":
                product_id = "suit_teal_coral"

            # Resolve background environment
            env_bgr = ENVIRONMENTS_CACHE.get(environment)

            # --- Delegate to the face-safe TryOnEngine ---------------------
            output_bgr = self._engine.process_frame(
                frame_bgr=frame_bgr,
                product_id=product_id,
                prompt=prompt,
                environment_bgr=env_bgr,
            )

            # --- Encode output ---------------------------------------------
            jpeg_bytes = encode_jpeg(output_bgr, quality=82)
            out_b64    = base64.b64encode(jpeg_bytes).decode("utf-8")
            latency_ms = (time.time() - start_time) * 1000
            return out_b64, latency_ms

        except Exception as e:
            logger.error(f"process_frame error: {e}", exc_info=True)
            return None, (time.time() - start_time) * 1000

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

        # Try Modal Cloud GPU Web Endpoint first
        modal_url = os.environ.get("MODAL_TRYON_URL", "https://rehmateramzantransmission--tvm-live-try-on-web-tryon.modal.run")
        try:
            import requests
            with open(garm_path, "rb") as gf:
                garm_b64 = base64.b64encode(gf.read()).decode("utf-8")
            resp = requests.post(modal_url, json={
                "user_image": frame_b64,
                "garment_image": garm_b64,
                "description": f"{product.brand} {product.name}"
            }, timeout=45)
            if resp.status_code == 200:
                res_data = resp.json()
                if res_data.get("status") == "success" and res_data.get("image"):
                    logger.info(f"⚡ Modal Cloud GPU HD Fit finished for product {product_id}!")
                    return res_data["image"]
        except Exception as modal_err:
            logger.info(f"Modal web endpoint fallback: {modal_err}")

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
