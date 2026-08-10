"""
tryon_engine.py
================
Real-time Virtual Try-On Engine
Architecture:
  1. MediaPipe Pose  → extract body landmarks → build TORSO polygon mask
  2. MediaPipe Face Detection / Face Mesh → build FACE protection mask
  3. Garment PNG warped via perspective transform onto torso quad
  4. Optional: StableDiffusionInpaintPipeline targeted STRICTLY at torso mask
              (seam / lighting harmonization only – never touches face pixels)
  5. Face pixels composited back 100% from raw camera frame → zero hallucination
"""

from __future__ import annotations

import base64
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger("tvm-server")

# ---------------------------------------------------------------------------
# Optional: PyTorch Inpaint Pipeline (Colab GPU only)
# ---------------------------------------------------------------------------
HAS_INPAINT = False
_inpaint_pipe = None

def init_inpaint_pipeline():
    """
    Load StableDiffusionInpaintPipeline with LCM-LoRA for fast GPU inpainting.
    Falls back silently on CPU / missing deps.
    """
    global HAS_INPAINT, _inpaint_pipe
    try:
        import torch
        from diffusers import StableDiffusionInpaintPipeline, LCMScheduler

        if not torch.cuda.is_available():
            logger.info("Inpaint pipeline: no CUDA → skipping (garment warp-only mode).")
            return

        logger.info("⚡ CUDA detected — loading SD Inpaint + LCM-LoRA for seam harmonization …")
        model_id = "runwayml/stable-diffusion-inpainting"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to("cuda")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        try:
            pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
            pipe.fuse_lora()
            logger.info("LCM-LoRA fused into inpaint pipeline.")
        except Exception as lora_err:
            logger.warning(f"LCM-LoRA load skipped: {lora_err}")

        _inpaint_pipe = pipe
        HAS_INPAINT = True
        logger.info("✅ SD Inpaint + LCM-LoRA ready — seam harmonization ACTIVE.")
    except Exception as e:
        logger.info(f"Inpaint pipeline fallback (warp-only): {e}")


# ---------------------------------------------------------------------------
# Pose model download helper
# ---------------------------------------------------------------------------
MODEL_DIR = Path("models")
POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker_lite.task"
FACE_MODEL_PATH = MODEL_DIR / "face_landmarker.task"

def ensure_models():
    MODEL_DIR.mkdir(exist_ok=True)

    if not POSE_MODEL_PATH.exists():
        url = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
               "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
        logger.info(f"Downloading pose model …")
        urllib.request.urlretrieve(url, str(POSE_MODEL_PATH))
        logger.info("Pose model downloaded ✓")

    if not FACE_MODEL_PATH.exists():
        url = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
               "face_landmarker/float16/latest/face_landmarker.task")
        logger.info("Downloading face landmarker model …")
        try:
            urllib.request.urlretrieve(url, str(FACE_MODEL_PATH))
            logger.info("Face landmarker model downloaded ✓")
        except Exception as e:
            logger.warning(f"Face landmarker download failed (will use pose nose/chin): {e}")


# ---------------------------------------------------------------------------
# TryOnEngine
# ---------------------------------------------------------------------------
class TryOnEngine:
    """
    Landmark-guided, face-safe virtual try-on engine.
    """

    # Fraction of frame height that is ALWAYS treated as face-safe zone
    # (used as fallback when face landmarks are unavailable)
    FACE_SAFE_FRACTION = 0.42  

    def __init__(self, garments_dir: Path, products: list):
        ensure_models()
        init_inpaint_pipeline()

        # ---- Pose landmarker (IMAGE mode, sync) ----------------------------
        pose_opts = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.45,
            min_tracking_confidence=0.45,
            output_segmentation_masks=True,
        )
        self.pose_lm = mp.tasks.vision.PoseLandmarker.create_from_options(pose_opts)

        # ---- Face landmarker (IMAGE mode) – optional -----------------------
        self.face_lm = None
        if FACE_MODEL_PATH.exists():
            try:
                face_opts = mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.4,
                )
                self.face_lm = mp.tasks.vision.FaceLandmarker.create_from_options(face_opts)
                logger.info("Face Landmarker initialised ✓")
            except Exception as e:
                logger.warning(f"Face landmarker init failed: {e}")

        # ---- Garment cache -------------------------------------------------
        self.garments: Dict[str, np.ndarray] = {}
        self._load_garments(garments_dir, products)

        self._log_count = 0
        logger.info("TryOnEngine ready — MediaPipe Pose + Face Protection + Garment Warp ✅")

    # ------------------------------------------------------------------
    # Garment loading
    # ------------------------------------------------------------------
    def _load_garments(self, garments_dir: Path, products: list):
        for product in products:
            path = garments_dir / f"{product.id}.png"
            if path.exists():
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    self.garments[product.id] = img
                    logger.info(f"Loaded garment: {product.id}  shape={img.shape}")

    def add_custom_garment(self, garment_id: str, img_b64: str,
                           garments_dir: Path) -> bool:
        try:
            if "," in img_b64:
                img_b64 = img_b64.split(",")[1]
            raw = np.frombuffer(base64.b64decode(img_b64), np.uint8)
            base_img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if base_img is None:
                return False

            w, h = 512, 640
            base_resized = cv2.resize(base_img, (w, h), interpolation=cv2.INTER_AREA)
            img = np.zeros((h, w, 4), dtype=np.uint8)
            img[:, :, :3] = base_resized

            # Build a simple shirt-silhouette alpha mask
            pts = np.array([
                [int(w*0.12), 0],  [int(w*0.38), 0],
                [int(w*0.50), int(h*0.08)],
                [int(w*0.62), 0],  [int(w*0.88), 0],
                [int(w*0.98), int(h*0.12)],
                [int(w*0.92), int(h*0.35)], [int(w*0.78), int(h*0.28)],
                [int(w*0.82), h],   [int(w*0.18), h],
                [int(w*0.22), int(h*0.28)], [int(w*0.08), int(h*0.35)],
                [int(w*0.02), int(h*0.12)],
            ], np.int32)
            alpha = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(alpha, [pts], 255)
            alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
            img[:, :, 3] = alpha

            out_path = garments_dir / f"{garment_id}.png"
            cv2.imwrite(str(out_path), img)
            self.garments[garment_id] = img
            logger.info(f"Custom garment {garment_id} added.")
            return True
        except Exception as e:
            logger.error(f"add_custom_garment error: {e}")
            return False

    # ------------------------------------------------------------------
    # Landmark helpers
    # ------------------------------------------------------------------
    def _get_face_bbox_pixels(self, rgb_frame: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
        """
        Returns (x1, y1, x2, y2) bounding box of face in pixel coords, or None.
        Uses FaceLandmarker if available, else falls back to Haarcascade.
        """
        h, w = rgb_frame.shape[:2]

        if self.face_lm is not None:
            try:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                res = self.face_lm.detect(mp_img)
                if res.face_landmarks and len(res.face_landmarks) > 0:
                    xs = [lm.x * w for lm in res.face_landmarks[0]]
                    ys = [lm.y * h for lm in res.face_landmarks[0]]
                    pad_x = int(0.08 * w)
                    pad_y = int(0.06 * h)
                    x1 = max(0, int(min(xs)) - pad_x)
                    y1 = max(0, int(min(ys)) - pad_y)
                    x2 = min(w, int(max(xs)) + pad_x)
                    y2 = min(h, int(max(ys)) + pad_y)
                    return x1, y1, x2, y2
            except Exception:
                pass

        # Haarcascade fallback
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            if len(faces) > 0:
                x, y, fw, fh = faces[0]
                pad = int(0.1 * max(fw, fh))
                return (max(0, x-pad), max(0, y-pad),
                        min(w, x+fw+pad), min(h, y+fh+pad))
        except Exception:
            pass
        return None

    def _build_torso_mask(
        self,
        landmarks,
        h: int, w: int,
        face_bottom_y: int,
    ) -> np.ndarray:
        """
        Build a smooth binary mask for the torso polygon
        (shoulders → hips) BELOW the face bounding box.
        Returns float32 [0..1] mask of shape (h, w).
        """
        PL = mp.tasks.vision.PoseLandmark

        l_sh = landmarks[PL.LEFT_SHOULDER]
        r_sh = landmarks[PL.RIGHT_SHOULDER]
        l_hp = landmarks[PL.LEFT_HIP]
        r_hp = landmarks[PL.RIGHT_HIP]

        shoulder_vis = l_sh.visibility > 0.25 and r_sh.visibility > 0.25
        hip_vis      = l_hp.visibility > 0.25 and r_hp.visibility > 0.25

        if not shoulder_vis:
            # Cannot build mask – return empty
            return np.zeros((h, w), dtype=np.float32)

        # Pixel coordinates (clamped)
        def px(lm): return (int(np.clip(lm.x * w, 0, w-1)),
                             int(np.clip(lm.y * h, 0, h-1)))

        ls_px = px(l_sh)
        rs_px = px(r_sh)

        shoulder_w_px = abs(ls_px[0] - rs_px[0])

        if hip_vis:
            lh_px = px(l_hp)
            rh_px = px(r_hp)
        else:
            # Estimate hips from shoulders
            mid_x = (ls_px[0] + rs_px[0]) // 2
            hip_y = min(h - 1, int(max(ls_px[1], rs_px[1]) + shoulder_w_px * 1.4))
            half_hw = int(shoulder_w_px * 0.56)
            lh_px = (mid_x + half_hw, hip_y)
            rh_px = (mid_x - half_hw, hip_y)

        # Add lateral margin to cover sleeves
        lat = int(shoulder_w_px * 0.28)
        top_y = max(face_bottom_y, min(ls_px[1], rs_px[1]) - int(shoulder_w_px * 0.12))

        poly = np.array([
            [rs_px[0] - lat, top_y],          # top-right
            [ls_px[0] + lat, top_y],          # top-left
            [lh_px[0] + lat, lh_px[1]],       # bottom-left
            [rh_px[0] - lat, rh_px[1]],       # bottom-right
        ], dtype=np.int32)

        # Clamp to frame
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return mask.astype(np.float32) / 255.0

    # ------------------------------------------------------------------
    # Garment warp
    # ------------------------------------------------------------------
    def _warp_garment(
        self,
        garment: np.ndarray,
        landmarks,
        h: int, w: int,
        face_bottom_y: int,
        prompt: str = "",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perspective-warp garment PNG onto body quad.
        Returns (warped_bgr [h,w,3], alpha_float [h,w]) – both clipped to face_bottom_y.
        """
        PL = mp.tasks.vision.PoseLandmark

        l_sh = landmarks[PL.LEFT_SHOULDER]
        r_sh = landmarks[PL.RIGHT_SHOULDER]
        l_hp = landmarks[PL.LEFT_HIP]
        r_hp = landmarks[PL.RIGHT_HIP]

        shoulder_vis = l_sh.visibility > 0.25 and r_sh.visibility > 0.25
        if not shoulder_vis:
            return np.zeros((h, w, 3), np.uint8), np.zeros((h, w), np.float32)

        def pxf(lm): return (float(np.clip(lm.x * w, 0, w-1)),
                              float(np.clip(lm.y * h, 0, h-1)))

        ls = pxf(l_sh)
        rs = pxf(r_sh)
        shoulder_w = abs(ls[0] - rs[0])

        hip_vis = l_hp.visibility > 0.25 and r_hp.visibility > 0.25
        if hip_vis:
            lh = pxf(l_hp)
            rh = pxf(r_hp)
        else:
            mid_x = (ls[0] + rs[0]) / 2
            hip_y = min(h - 1.0, max(ls[1], rs[1]) + shoulder_w * 1.4)
            half = shoulder_w * 0.56
            lh = (mid_x + half, hip_y)
            rh = (mid_x - half, hip_y)

        lat = shoulder_w * 0.30
        top_y = max(float(face_bottom_y), min(ls[1], rs[1]) - shoulder_w * 0.10)

        dst = np.array([
            [rs[0] - lat, top_y],
            [ls[0] + lat, top_y],
            [lh[0] + lat, lh[1]],
            [rh[0] - lat, rh[1]],
        ], dtype=np.float32)

        g = garment.copy()

        # Prompt-guided tint
        if prompt:
            pl = prompt.lower()
            tint = None
            if   any(k in pl for k in ("red","maroon","ruby")):   tint=(30,30,200)
            elif any(k in pl for k in ("green","emerald")):        tint=(50,180,50)
            elif any(k in pl for k in ("gold","yellow","amber")):  tint=(30,215,255)
            elif any(k in pl for k in ("blue","teal","cyan")):     tint=(200,180,40)
            elif any(k in pl for k in ("purple","velvet")):        tint=(180,40,140)
            if tint:
                rgb = g[:,:,:3].astype(np.float32)
                ta  = np.full_like(rgb, tint, dtype=np.float32)
                g[:,:,:3] = np.clip(cv2.addWeighted(rgb,0.60,ta,0.40,0),0,255).astype(np.uint8)

        gh, gw = g.shape[:2]
        src = np.array([[0,0],[gw-1,0],[gw-1,gh-1],[0,gh-1]], dtype=np.float32)
        M   = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(g, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0,0,0,0))

        if warped.shape[2] == 4:
            alpha = warped[:,:,3].astype(np.float32) / 255.0
        else:
            alpha = np.ones((h, w), dtype=np.float32)

        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        alpha = np.clip(alpha * 1.12, 0.0, 1.0)

        # Hard-zero above face bottom line
        alpha[:face_bottom_y, :] = 0.0

        return warped[:,:,:3], alpha

    # ------------------------------------------------------------------
    # Inpaint seam harmonization (GPU only)
    # ------------------------------------------------------------------
    def _inpaint_seam(
        self,
        composited: np.ndarray,   # BGR [h,w,3] after garment warp blend
        torso_mask: np.ndarray,   # float32 [h,w] 0..1
        product_id: str,
        warped_bgr: np.ndarray,   # original garment warp BGR – used as fallback
        alpha: np.ndarray,        # garment alpha float32 [h,w] – used as fallback
    ) -> np.ndarray:
        """
        Run SD Inpaint ONLY on the thin seam border (dilate-erode ring) of the
        torso mask to harmonize lighting and garment edges.

        Safety guarantees:
          1. BGR→RGB conversion before sending to PIL / diffusion pipeline.
          2. Validates result is not all-black (mean pixel < 10 in masked area).
          3. If inpaint is dark/corrupt, falls back to the clean alpha-blended
             garment warp (composited) – NO BLACK BOX.
          4. Inpaint result composited only on the seam ring, not the full frame.
        """
        if not HAS_INPAINT or _inpaint_pipe is None:
            return composited
        try:
            import torch
            from PIL import Image

            # 1. Build narrow seam-border mask (dilated ring around garment)
            m_uint8 = (torso_mask * 255).astype(np.uint8)
            kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            dilated = cv2.dilate(m_uint8, kernel, iterations=1)
            eroded  = cv2.erode(m_uint8,  kernel, iterations=2)
            border  = cv2.subtract(dilated, eroded)
            border  = cv2.GaussianBlur(border, (9, 9), 0)

            if border.max() < 10:
                return composited   # no meaningful seam to process

            # 2. Convert BGR→RGB, ensure uint8 [0,255] for PIL  ← KEY FIX
            composited_u8 = np.clip(composited, 0, 255).astype(np.uint8)
            rgb_arr   = cv2.cvtColor(composited_u8, cv2.COLOR_BGR2RGB)  # RGB!
            pil_image = Image.fromarray(rgb_arr)          # RGB uint8 PIL
            pil_mask  = Image.fromarray(border.astype(np.uint8))  # L uint8

            # Resize to standard 512×512 the pipeline expects
            pil_image = pil_image.resize((512, 512), Image.LANCZOS)
            pil_mask  = pil_mask.resize((512, 512), Image.NEAREST)

            prompt_text = (
                f"photorealistic person wearing {product_id} jacket, "
                "sharp fabric texture, realistic studio lighting, seamless fit, 4k"
            )
            neg_prompt = (
                "black, dark, shadow, blurry, distorted, watermark, bad anatomy, "
                "extra limbs, disfigured"
            )

            # 3. Run SD Inpaint inference
            with torch.inference_mode():
                result_pil = _inpaint_pipe(
                    prompt=prompt_text,
                    negative_prompt=neg_prompt,
                    image=pil_image,
                    mask_image=pil_mask,
                    num_inference_steps=4,
                    guidance_scale=1.8,
                    strength=0.40,
                ).images[0]

            result_rgb = np.array(result_pil, dtype=np.uint8)  # RGB uint8

            # Resize back to original frame size if necessary
            h_orig, w_orig = composited.shape[:2]
            if result_rgb.shape[:2] != (h_orig, w_orig):
                result_rgb = cv2.resize(
                    result_rgb, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR
                )

            # 4. Black-output detection: check mean pixel inside mask region
            border_bin = (border > 30).astype(bool)
            if border_bin.any():
                masked_mean = float(result_rgb[border_bin].mean())
            else:
                masked_mean = 128.0

            if masked_mean < 10.0:
                # FALLBACK: inpaint returned dark/black → skip, keep warp
                logger.debug(
                    f"Inpaint dark output (mean={masked_mean:.1f}) → garment warp fallback"
                )
                return composited

            # 5. Composite inpaint result ONLY on the seam border ring
            #    (keeps clean garment interior, avoids full-frame replacement)
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            if result_bgr.shape[:2] != (h_orig, w_orig):
                result_bgr = cv2.resize(result_bgr, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

            border_f = border.astype(np.float32) / 255.0
            border3  = np.dstack([border_f] * 3)
            final    = (
                result_bgr.astype(np.float32) * border3
                + composited_u8.astype(np.float32) * (1.0 - border3)
            ).astype(np.uint8)
            return final

        except Exception as e:
            logger.debug(f"Inpaint seam error → warp fallback: {e}")
            return composited


    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def process_frame(
        self,
        frame_bgr: np.ndarray,
        product_id: str,
        prompt: str = "",
        environment_bgr: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Process a single BGR camera frame and return an output BGR frame
        with the garment applied to the torso while preserving the face 100%.

        Args:
            frame_bgr:        Raw camera frame (BGR)
            product_id:       Selected garment key
            prompt:           Optional free-form text for colour/tint guidance
            environment_bgr:  Optional background replacement image (BGR)
        Returns:
            output BGR frame same size as input
        """
        raw_frame = frame_bgr.copy()   # <-- pristine original, never modified
        h, w = frame_bgr.shape[:2]

        # ---- 1. Background swap (pose segmentation mask) ------------------
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        pose_res  = self.pose_lm.detect(mp_image)

        output = frame_bgr.copy()

        if (environment_bgr is not None
                and pose_res.segmentation_masks
                and len(pose_res.segmentation_masks) > 0):
            bg = cv2.resize(environment_bgr, (w, h))
            seg = pose_res.segmentation_masks[0].numpy_view().copy()
            if seg.shape[:2] != (h, w):
                seg = cv2.resize(seg, (w, h))
            seg = cv2.GaussianBlur(seg, (11, 11), 0)
            s3 = np.dstack([seg]*3)
            output = (output.astype(np.float32)*s3 + bg.astype(np.float32)*(1-s3)).astype(np.uint8)

        # ---- 2. Detect face bounding box (pixels) -------------------------
        face_bbox = self._get_face_bbox_pixels(rgb_frame)
        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = face_bbox
            face_bottom_y = fy2
        else:
            # Safe fallback: treat top FACE_SAFE_FRACTION as face zone
            face_bottom_y = int(h * self.FACE_SAFE_FRACTION)
            fx1, fy1, fx2, fy2 = 0, 0, w, face_bottom_y

        # ---- 3. Special shaders (non-garment modes) -----------------------
        product_lower = product_id.lower()
        prompt_lower  = prompt.lower()

        if product_id == "animated" or "animated" in prompt_lower or "pixar" in prompt_lower:
            color  = cv2.bilateralFilter(output, d=9, sigmaColor=75, sigmaSpace=75)
            gray   = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            blur   = cv2.medianBlur(gray, 7)
            edges  = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 9, 9)
            edges3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cartoon = cv2.bitwise_and(color, edges3)
            output  = cv2.addWeighted(cartoon, 0.82,
                                      np.full_like(cartoon, (30,80,140)), 0.18, 0)

        elif "alien" in product_lower or "alien" in prompt_lower:
            hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV).astype(int)
            hsv[:,:,0] = (hsv[:,:,0] + 35) % 180
            alien  = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            output = cv2.addWeighted(output, 0.3, alien, 0.7, 0)

        elif "slime" in product_lower or "slime" in prompt_lower:
            slime = np.zeros_like(output)
            slime[:,:,1] = 230; slime[:,:,0] = 50
            output = cv2.addWeighted(output, 0.65, slime, 0.35, 0)

        elif "fire" in product_lower or "fire" in prompt_lower:
            hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV).astype(int)
            hsv[:,:,0] = (hsv[:,:,0] + 160) % 180
            fire   = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            output = cv2.addWeighted(output, 0.5, fire, 0.5, 0)

        # ---- 4. Garment try-on (pose landmarks required) ------------------
        has_pose    = (pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0)
        has_garment = product_id in self.garments

        if has_pose and has_garment:
            landmarks = pose_res.pose_landmarks[0]

            # 4a. Build torso mask (used later for inpaint border)
            torso_mask = self._build_torso_mask(landmarks, h, w, face_bottom_y)

            # 4b. Warp garment PNG onto body quad
            warped_bgr, alpha = self._warp_garment(
                self.garments[product_id], landmarks, h, w, face_bottom_y, prompt
            )

            # 4c. Lighting-adaptive shading
            gray_f   = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            gray_b   = cv2.GaussianBlur(gray_f, (21, 21), 0)
            shading  = np.clip(gray_b * 0.55 + 0.55, 0.60, 1.30)
            shading3 = np.dstack([shading]*3)

            w_shaded = np.clip(warped_bgr.astype(np.float32) * shading3, 0, 255)
            a3 = np.dstack([alpha]*3)

            # 4d. Composite garment onto output
            blended = (w_shaded * a3 + output.astype(np.float32) * (1.0 - a3)).astype(np.uint8)

            # 4e. Optional: SD Inpaint seam harmonization on GPU
            blended = self._inpaint_seam(
                blended, torso_mask, product_id, warped_bgr, alpha
            )

            output = blended

        # ---- 5. HARD face restore ----------------------------------------
        # Whatever diffusion / compositing did, paste raw face pixels back.
        # This guarantees ZERO facial morphing.
        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = face_bbox
            # Expand face restore zone slightly (ears, hair)
            restore_y1 = max(0, fy1 - int(0.05*h))
            restore_y2 = min(h, fy2 + int(0.03*h))
            restore_x1 = max(0, fx1 - int(0.04*w))
            restore_x2 = min(w, fx2 + int(0.04*w))

            # Soft-feather blend at border
            face_roi_out = output[restore_y1:restore_y2, restore_x1:restore_x2]
            face_roi_raw = raw_frame[restore_y1:restore_y2, restore_x1:restore_x2]

            rh_ = restore_y2 - restore_y1
            rw_ = restore_x2 - restore_x1
            # Feather mask: full-opaque centre, soft edges
            feat = np.zeros((rh_, rw_), dtype=np.float32)
            inner_y = max(1, rh_//6); inner_x = max(1, rw_//6)
            feat[inner_y:-inner_y, inner_x:-inner_x] = 1.0
            feat = cv2.GaussianBlur(feat, (int(min(rh_,rw_)//4)*2+1,
                                           int(min(rh_,rw_)//4)*2+1), 0)
            feat3 = np.dstack([feat]*3)

            restored = (face_roi_raw.astype(np.float32)*feat3
                        + face_roi_out.astype(np.float32)*(1-feat3)).astype(np.uint8)
            output[restore_y1:restore_y2, restore_x1:restore_x2] = restored
        else:
            # Fallback: always restore top FACE_SAFE_FRACTION rows
            output[:face_bottom_y, :, :] = raw_frame[:face_bottom_y, :, :]

        return output
