# 🚀 Camstream-2 Quick Start Guide

## 📋 Project Overview
**Camstream-2** is a Rust-driven real-time camera analyzer that uses Apple's FastVLM AI model to describe what's happening in your camera feed in real-time.

## ⚡ Quick Setup (5 minutes)

### 1. Prerequisites
```bash
# Install FFmpeg (if not already installed)
brew install ffmpeg

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### 2. Clone & Setup
```bash
# Clone the repository
git clone https://github.com/mvillafranca98/Camstream-2.git
cd Camstream-2

# Run automated setup
chmod +x setup.sh
./setup.sh
```

### 3. Run the Camera Analyzer
```bash
# Activate virtual environment
source .venv/bin/activate

# Start the camera analyzer
./target/debug/camstream
```

## 🎯 What You'll See
```
🚀 Rust-driven Camera Stream Analyzer
=====================================
🤖 Python FastVLM module initialized by Rust
📹 Camera settings: 640x480 @ 30 FPS
📊 Processing every 15 frames (analysis rate: 2 FPS)
✅ Using Apple MPS backend
🤖 Loading model: apple/FastVLM-0.5B
✅ FastVLM model loaded successfully

[Frame 0] [t=8.5s] [fps=0.0] In the image, a man is captured in a moment of tranquility...
[Frame 1] [t=10.8s] [fps=0.1] In the image, a man with curly hair and a beard...
```

## 🛠️ Manual Setup (Alternative)

### 1. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Build Project
```bash
# Build PyO3 module
maturin develop

# Build Rust binary
cargo build --bin camstream
```

### 4. Test Setup
```bash
python test_setup.py
```

## 🎮 Usage Modes

### Mode 1: Rust-Driven (Recommended)
```bash
./target/debug/camstream
```
- Rust orchestrates everything
- Real-time output streaming
- Best performance

### Mode 2: Python-Only
```bash
source .venv/bin/activate
python live_fastvlm_enhanced.py
```
- Direct Python execution
- More configuration options

### Mode 3: Testing
```bash
source .venv/bin/activate
python test_mac_camera.py
```
- Test camera framerates
- Debug camera issues

## ⚙️ Configuration

Edit `config.py` to customize:
```python
CAMERA_CONFIG = {
    "device": "0",           # Camera device index
    "width": 640,            # Resolution width
    "height": 480,           # Resolution height
    "framerate": 30,         # FPS
}

PROCESSING_CONFIG = {
    "max_frames": 50,        # Process 50 frames then stop
    "question": "Describe this scene in detail",
    "max_new_tokens": 128,   # Description length
}
```

## 🔧 Troubleshooting

### Camera Issues
```bash
# List available cameras
ffmpeg -f avfoundation -list_devices true -i ""

# Grant camera permissions in System Preferences > Security & Privacy
```

### Build Issues
```bash
# Clean and rebuild
cargo clean
maturin develop
cargo build --bin camstream
```

### Python Issues
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

## 📁 Project Structure
```
Camstream-2/
├── src/
│   ├── main_simple.rs      # Rust binary (main entry point)
│   ├── main.rs             # Alternative PyO3 integration
│   └── lib.rs              # PyO3 module for camera capture
├── fastvlm_analyzer.py     # FastVLM AI analysis module
├── rust_camera_cli.py      # Python CLI called by Rust
├── live_fastvlm_enhanced.py # Enhanced Python version
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── setup.sh               # Automated setup script
├── Cargo.toml             # Rust project configuration
└── target/debug/camstream # Built Rust binary
```

## 🎯 Key Commands

### Development
```bash
# Build Rust binary
cargo build --bin camstream

# Run tests
python test_setup.py

# Clean build
cargo clean
```

### Git Operations
```bash
# Pull latest changes
git pull origin main

# Push changes
git add .
git commit -m "Your message"
git push origin main
```

## 🚨 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "Camera permission denied" | Grant camera access in System Preferences |
| "FFmpeg not found" | Run `brew install ffmpeg` |
| "Module not found" | Activate virtual environment: `source .venv/bin/activate` |
| "Build failed" | Run `maturin develop` then `cargo build --bin camstream` |
| "No camera devices" | Check camera connections and permissions |

## 📊 Performance Tips

### For Better Performance
- Use lower resolution (320x240)
- Increase frame skip (30 for 1 FPS analysis)
- Close other applications

### For Better Quality
- Use higher resolution (1280x720)
- Decrease frame skip (15 for 2 FPS analysis)
- Increase max_new_tokens for longer descriptions

## 🎉 Success Indicators

✅ **Setup Complete When:**
- `./target/debug/camstream` runs without errors
- Camera turns on and shows live feed
- AI descriptions appear in real-time
- No permission or module errors

✅ **Working Correctly When:**
- You see frame descriptions like "In the image, a person is..."
- FPS counter shows processing rate
- Timestamps increase over time
- Ctrl+C stops gracefully

## 📞 Support

- **Repository**: https://github.com/mvillafranca98/Camstream-2
- **Issues**: Create an issue on GitHub
- **Documentation**: See README.md for full details

---
*Last updated: $(date)*
*Project: Camstream-2 - Rust-driven FastVLM Camera Analyzer*
