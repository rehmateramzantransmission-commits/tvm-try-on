# =====================================================================
# TVM Smart Mirror — 1-Click Standalone Self-Contained Colab Launcher
# Zero GitHub / Git dependencies needed!
# =====================================================================

import os
import sys
import base64
import subprocess

# Credentials
HF_TOKEN = ""
NGROK_TOKEN = "3HYHgO7KK7jrszy6w9zAkHV2aRa_255eksu45pewDkpcP6Bk"

os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
os.environ["NGROK_AUTH_TOKEN"] = NGROK_TOKEN

print("📦 Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn[standard]", "websockets", "opencv-python-headless", "mediapipe", "numpy", "pillow", "pyngrok", "gradio_client"], check=True)

# Create workspace structure
os.makedirs("static/products/garments", exist_ok=True)
os.makedirs("static/environments", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Read server.py and index.html from local path if exists, or embedded
with open("server.py", "r", encoding="utf-8") as f:
    server_code = f.read()

from pyngrok import ngrok
ngrok.set_auth_token(NGROK_TOKEN)

try:
    ngrok.kill()
    tunnel = ngrok.connect(8000)
    public_url = tunnel.public_url
    print("\n" + "★" * 65)
    print(f"✨ PUBLIC APP URL: {public_url}")
    print(f"✨ WEBSOCKET URL:  {public_url.replace('http://', 'ws://').replace('https://', 'wss://')}/ws/tryon")
    print("★" * 65 + "\n")
except Exception as e:
    print(f"Ngrok notice: {e}")
    public_url = "http://localhost:8000"

print("🔥 Launching FastAPI AI Server...")
subprocess.run([sys.executable, "server.py", "--port", "8000", "--no-tunnel"])
