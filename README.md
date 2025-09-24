# 🎥 Live FastVLM Camera Stream Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

A **Rust-driven** real-time camera stream analyzer that captures video from your camera using ffmpeg via PyO3 and analyzes each frame using Apple's FastVLM model to describe what's happening in real-time.

## ✨ Features

- 🦀 **Rust-driven architecture** - Rust binary orchestrates the entire pipeline
- 🎥 **Real-time camera capture** using ffmpeg via PyO3
- 🤖 **FastVLM integration** for scene analysis
- 📊 **Performance monitoring** with FPS and timing information
- 🛡️ **Robust error handling** with graceful shutdown
- ⚙️ **Configurable settings** via config.py
- 📁 **Frame saving** capability for debugging
- 🔍 **Auto device detection** for camera selection
- 🔄 **Multiple execution modes** - Rust-driven or Python-driven
- 🍎 **macOS optimized** with Apple MPS support

## Requirements

### System Requirements
- macOS (uses AVFoundation for camera access)
- Python 3.8+
- FFmpeg installed and in PATH
- Rust toolchain (for building PyO3 module)

### Python Dependencies
- torch
- PIL (Pillow)
- transformers
- maturin (for building PyO3 module)

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/camstream.git
cd camstream

# Run the setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment and run
source .venv/bin/activate
./target/debug/camstream
```

### Option 2: Manual Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/camstream.git
cd camstream

# Install dependencies
pip install -r requirements.txt

# Build the project
maturin develop
cargo build --bin camstream

# Run the camera analyzer
./target/debug/camstream
```

## 📋 Installation

### Prerequisites

1. **Install FFmpeg** (if not already installed):
   ```bash
   brew install ffmpeg
   ```

2. **Install Rust** (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

### Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Build the PyO3 module and Rust binary
maturin develop
cargo build --bin camstream
```

### Verify Installation

```bash
# Test your setup
python test_setup.py
```

## Usage

### 🦀 Rust-Driven Mode (Recommended)

Run the Rust binary that orchestrates the entire pipeline:
```bash
./target/debug/camstream
```

This is the **true Rust-driven approach** where:
- Rust is the main application
- Rust controls camera capture and AI analysis
- Python FastVLM runs as a subprocess
- Real-time output streaming

### 🐍 Python-Driven Mode

**Enhanced version with configuration:**
```bash
source .venv/bin/activate
python live_fastvlm_enhanced.py
```

**Original version:**
```bash
source .venv/bin/activate
python live_fastvlm.py
```

### 🧪 Testing Mode

**Test camera framerates:**
```bash
source .venv/bin/activate
python test_mac_camera.py
```

### Configuration

Edit `config.py` to customize:

- **Camera settings**: resolution, framerate, device selection
- **Processing settings**: max frames, analysis question
- **Output settings**: timestamps, FPS display, frame saving
- **Advanced settings**: error recovery, validation

### Example Configuration

```python
# config.py
CAMERA_CONFIG = {
    "device": "0",           # Camera device index
    "width": 1280,           # Higher resolution
    "height": 720,
    "framerate": 5,          # Higher framerate
}

PROCESSING_CONFIG = {
    "max_frames": 50,        # Process 50 frames then stop
    "question": "Describe this scene in detail",
    "max_new_tokens": 128,   # Longer descriptions
}
```

## Controls

- **Ctrl+C**: Graceful shutdown
- **SIGTERM**: Graceful shutdown (if configured)

## Output Format

```
[Frame 0] [t=0.5s] [fps=2.1] A person is sitting at a desk working on a computer
[Frame 1] [t=1.0s] [fps=2.0] The person is typing on the keyboard
```

Where:
- `Frame X`: Frame number
- `t=X.Xs`: Timestamp from capture start
- `fps=X.X`: Current processing FPS
- Description: FastVLM's analysis of the frame

## Project Structure

```
camstream/
├── src/
│   ├── lib.rs              # PyO3 module for camera capture
│   ├── main.rs             # Full PyO3 integration (advanced)
│   └── main_simple.rs      # Rust binary (current implementation)
├── live_fastvlm.py         # Original Python script
├── live_fastvlm_enhanced.py # Enhanced Python script with config
├── rust_camera_cli.py      # Python CLI for Rust binary
├── fastvlm_analyzer.py     # FastVLM analysis module
├── config.py               # Configuration settings
├── test_setup.py           # Setup verification
├── test_camera.py          # Camera testing
├── test_mac_camera.py      # Mac-specific camera testing
├── Cargo.toml              # Rust project configuration
├── pyproject.toml          # Python project configuration
└── target/debug/camstream  # Built Rust binary
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**:
   - Install with `brew install ffmpeg`
   - Ensure it's in your PATH

2. **Camera permission denied**:
   - Grant camera permissions in System Preferences > Security & Privacy
   - Try different device indices

3. **PyO3 module build fails**:
   - Ensure Rust is installed: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
   - Run `maturin develop` from the project directory

4. **Rust binary not found**:
   - Build with `cargo build --bin camstream`
   - Ensure you're in the project directory

5. **No camera devices found**:
   - Check camera connections
   - Run `ffmpeg -f avfoundation -list_devices true -i ""` to list devices
   - Grant camera permissions to Terminal

6. **Python module not found in Rust mode**:
   - Ensure you're using the virtual environment Python
   - Run `maturin develop` to install the camstream module

### Debug Mode

Enable verbose logging in `config.py`:
```python
ADVANCED_CONFIG = {
    "verbose_logging": True,
    "frame_validation": True,
    "error_recovery": True,
}
```

### Frame Saving

To save frames for debugging:
```python
OUTPUT_CONFIG = {
    "save_frames": True,
    "output_dir": "debug_frames",
}
```

## Architecture

### 🦀 Rust-Driven Architecture (Recommended)
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Rust Binary   │───▶│   Python CLI     │───▶│   FastVLM       │
│   (Main App)    │    │   (Interface)    │    │   (AI Analysis) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Camera        │    │   PyO3 Module    │    │   Results       │
│   (FFmpeg)      │◀───│   (Rust)         │◀───│   (Back to Rust)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 🐍 Python-Driven Architecture (Alternative)
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Camera        │    │   PyO3 Module    │    │   FastVLM       │
│   (FFmpeg)      │───▶│   (Rust)         │───▶│   (Python)      │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

**Key Components:**
1. **FFmpeg** captures raw RGB frames from camera
2. **PyO3 module** provides efficient frame reading and validation
3. **FastVLM** analyzes each frame and generates descriptions
4. **Rust Binary** (in Rust-driven mode) orchestrates the entire pipeline

## Performance Tips

### 🚀 Optimization Strategies

**For Rust-driven mode:**
- Adjust frame skip in `rust_camera_cli.py` arguments
- Modify max_frames parameter for different session lengths
- Use Rust's native logging for better performance monitoring

**For Python-driven mode:**
- Lower resolution for faster processing
- Reduce framerate for more stable analysis
- Use MPS (Apple Silicon) or CUDA for GPU acceleration
- Adjust `max_new_tokens` for shorter/longer descriptions

### 🎯 Recommended Settings

**Mac Built-in Camera:**
- Framerate: 30 FPS (required by macOS)
- Frame skip: 15 (for 2 FPS analysis)
- Resolution: 640x480 (good balance of speed/quality)

**High Performance:**
- Framerate: 30 FPS
- Frame skip: 30 (for 1 FPS analysis)
- Resolution: 320x240

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note:** This project uses Apple's FastVLM model. Please check their license terms for usage restrictions.

## 🙏 Acknowledgments

- [Apple's FastVLM](https://huggingface.co/apple/FastVLM-0.5B) for the vision-language model
- [PyO3](https://pyo3.rs/) for Python-Rust bindings
- [FFmpeg](https://ffmpeg.org/) for camera capture
- [Hugging Face Transformers](https://huggingface.co/transformers/) for model loading
