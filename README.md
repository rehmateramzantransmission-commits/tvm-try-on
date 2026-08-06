# TVM Smart Mirror — 1-Click Google Colab Guide

Run the full **Decart Lucy AI World Model** & **Photorealistic Virtual Try-On** engine on Google Colab's Free T4 GPU.

---

## ⚡ 1-Click Execution Procedure (No GitHub / Password Required!)

### Step 1: Open Google Colab
Open **[colab.research.google.com](https://colab.research.google.com/)** and click **New Notebook**.

### Step 2: Enable Free T4 GPU
Click **Runtime ➔ Change runtime type ➔ Select T4 GPU ➔ Click Save**.

### Step 3: Run the 1-Cell Installer

Copy and paste the python code from [`colab_one_cell_script.py`](file:///c:/Users/Muhammad%20Shoaib/Documents/projects/tvm-live-try-on/colab_one_cell_script.py) into your Colab notebook cell and press **Play (▶)**.

*(Alternatively: Upload `colab_one_cell_script.py` directly to the Colab Files panel on the left and run `!python colab_one_cell_script.py`)*

### Step 4: Open Public App URL
When execution completes, Colab will print:
```
★ =================================================================
✨ PUBLIC APP URL: https://xxxx.ngrok-free.app
★ =================================================================
```
Click that URL in **Google Chrome**, allow camera access, and click **Start Mirror**!

---

## 🌟 What You Can Do in the App:
1. 🏬 **World Environment Editing**: Change background to *Khaadi Boutique*, *Fashion Runway*, *Royal Palace*, or *Studio Box*.
2. 🎨 **AI Prompt Styling**: Type prompts like `#Red Velvet`, `#Emerald Green`, or `#Gold Embroidered` to transform outfit colors live.
3. 📁 **Upload Outfit**: Click `+ Upload Outfit` to try on any clothing photo from your device.
4. ✨ **HD 2.5D AI Fit**: Click `✨ GENERATE HD 2.5D AI FIT` for 1-click **IDM-VTON Photorealistic Neural Fitting**.
