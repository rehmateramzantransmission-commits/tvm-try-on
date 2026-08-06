# =====================================================================
# TVM Smart Mirror & World Model Engine — Google Colab 1-Click Launcher
# HuggingFace & Ngrok tokens pre-configured for instant execution
# =====================================================================

import os
import sys
import subprocess
import time

# User Provided Credentials
HF_TOKEN = ""
NGROK_TOKEN = "3HYHgO7KK7jrszy6w9zAkHV2aRa_255eksu45pewDkpcP6Bk"

print("=" * 60)
print("🚀 Starting TVM Smart Mirror & Decart Lucy AI Engine on Colab GPU")
print("=" * 60)

# Set environment variables
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
os.environ["NGROK_AUTH_TOKEN"] = NGROK_TOKEN

# Install system dependencies & pyngrok
print("📦 Checking & installing required packages...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyngrok", "gradio_client", "fastapi", "uvicorn[standard]", "websockets", "mediapipe", "opencv-python-headless", "pillow", "numpy", "diffusers", "transformers", "accelerate", "peft", "fal-client"], check=True)

from pyngrok import ngrok

# Set ngrok auth token
ngrok.set_auth_token(NGROK_TOKEN)

# Start ngrok tunnel on port 8000
print("🌍 Launching Ngrok Secure Tunnel...")
try:
    ngrok.kill()  # kill any stale tunnels
    tunnel = ngrok.connect(8000)
    public_url = tunnel.public_url
    print("\n" + "★" * 60)
    print(f"✨ PUBLIC APP URL: {public_url}")
    print(f"✨ WEBSOCKET URL:  {public_url.replace('http://', 'ws://').replace('https://', 'wss://')}/ws/tryon")
    print("★" * 60 + "\n")
except Exception as e:
    print(f"⚠️ Ngrok tunnel notice: {e}")
    public_url = "http://localhost:8000"

print("🔥 Launching FastAPI Server...")
cmd = [sys.executable, "server.py", "--port", "8000", "--no-tunnel"]
subprocess.run(cmd)
