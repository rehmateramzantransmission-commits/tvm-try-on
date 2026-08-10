"""
modal_tryon.py
================
Modal Cloud GPU Virtual Try-On Serverless Backend
Runs IDM-VTON / High-Quality Try-On on NVIDIA GPUs (A10G / T4)
"""

import base64
import os
from pathlib import Path
import modal

# Define container image with all PyTorch & CV dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "opencv-python-headless",
        "pillow",
        "numpy",
        "mediapipe",
        "gradio_client",
        "fastapi",
        "uvicorn"
    )
)

app = modal.App("tvm-live-try-on", image=image)

@app.cls(gpu="A10G", timeout=300, min_containers=0)
class TryOnServer:
    @modal.enter()
    def setup(self):
        print("⚡ Modal GPU Instance Booting... Loading Try-On Models...")
        from gradio_client import Client
        try:
            self.client = Client("yisol/IDM-VTON")
            print("✅ IDM-VTON Client Initialized on Modal GPU!")
        except Exception as e:
            print(f"IDM-VTON client init warning: {e}")
            self.client = None

    @modal.method()
    def predict_hd(self, user_b64: str, garment_b64: str, garment_desc: str = "Tuxedo Suit") -> str:
        """
        Runs neural virtual try-on on GPU.
        Input: user_b64 (camera photo), garment_b64 (clothing asset)
        Output: Base64 result image with photorealistic apparel fit
        """
        import tempfile
        from gradio_client import handle_file

        with tempfile.TemporaryDirectory() as tmpdir:
            user_path = os.path.join(tmpdir, "user.jpg")
            garm_path = os.path.join(tmpdir, "garment.png")

            if "," in user_b64:
                user_b64 = user_b64.split(",")[1]
            with open(user_path, "wb") as f:
                f.write(base64.b64decode(user_b64))

            if "," in garment_b64:
                garment_b64 = garment_b64.split(",")[1]
            with open(garm_path, "wb") as f:
                f.write(base64.b64decode(garment_b64))

            print(f"🚀 Running IDM-VTON GPU Inference for: {garment_desc}...")
            
            if self.client is None:
                from gradio_client import Client
                self.client = Client("yisol/IDM-VTON")

            res = self.client.predict(
                dict={"background": handle_file(user_path), "layers": [], "composite": None},
                garm_img=handle_file(garm_path),
                garment_des=garment_desc,
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=25,
                seed=42,
                api_name="/tryon"
            )

            out_file = res[0]
            with open(out_file, "rb") as f:
                result_bytes = f.read()

            out_b64 = base64.b64encode(result_bytes).decode("utf-8")
            print("✅ IDM-VTON GPU Inference Finished Successfully!")
            return out_b64

@app.function()
@modal.fastapi_endpoint(method="POST")
def web_tryon(item: dict):
    """
    HTTP POST Webhook Endpoint for instant frontend integration
    """
    user_b64 = item.get("user_image", "")
    garment_b64 = item.get("garment_image", "")
    desc = item.get("description", "Tuxedo Suit")

    server = TryOnServer()
    result_b64 = server.predict_hd.remote(user_b64, garment_b64, desc)
    return {"status": "success", "image": result_b64}
