# 🏗️ Building a Rust-Python Camera AI Analyzer from Scratch

A complete step-by-step guide to create a project like Camstream-2 from the ground up.

## 📋 Project Overview

We'll build a real-time camera analyzer that:
- Captures video using FFmpeg
- Uses Rust for performance-critical parts
- Integrates Python AI models via PyO3
- Provides real-time scene descriptions

## 🎯 Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Rust Binary   │───▶│   Python CLI     │───▶│   FastVLM AI    │
│   (Main App)    │    │   (Interface)    │    │   (Analysis)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Camera        │    │   PyO3 Module    │    │   Results       │
│   (FFmpeg)      │◀───│   (Rust)         │◀───│   (Back to Rust)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Step 1: Project Setup

### 1.1 Create Project Structure
```bash
mkdir camstream-project
cd camstream-project

# Create directory structure
mkdir src
mkdir tests
touch README.md
touch .gitignore
```

### 1.2 Initialize Rust Project
```bash
# Initialize Cargo project
cargo init --name camstream

# Add PyO3 dependencies to Cargo.toml
cat >> Cargo.toml << 'EOF'
[lib]
name = "camstream"
crate-type = ["cdylib"]

[[bin]]
name = "camstream"
path = "src/main_simple.rs"

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
anyhow = "1.0"
thiserror = "1.0"
ctrlc = "3.4"
EOF
```

### 1.3 Initialize Python Project
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Create Python project files
touch pyproject.toml
touch requirements.txt
```

## 🐍 Step 2: Python AI Module

### 2.1 Create FastVLM Analyzer
```python
# fastvlm_analyzer.py
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

# Global variables for caching
_model = None
_tokenizer = None
_device = None
_dtype = None

def initialize_model():
    """Initialize the FastVLM model and tokenizer"""
    global _model, _tokenizer, _device, _dtype
    
    if _model is not None:
        return
    
    MODEL_ID = "apple/FastVLM-0.5B"
    
    # Detect device
    if torch.backends.mps.is_available():
        _device = torch.device("mps")
        _dtype = torch.float16
    elif torch.cuda.is_available():
        _device = torch.device("cuda")
        _dtype = torch.float16
    else:
        _device = torch.device("cpu")
        _dtype = torch.float32
    
    # Load model and tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=_dtype,
        device_map=None,
        trust_remote_code=True,
    )
    _model.to(_device)

def analyze_frame(frame_bytes, width, height):
    """Analyze a frame using FastVLM"""
    global _model, _tokenizer, _device, _dtype
    
    if _model is None:
        initialize_model()
    
    try:
        # Convert bytes to PIL Image
        img = Image.frombytes("RGB", (width, height), frame_bytes)
        
        # Preprocess image
        px = _model.get_vision_tower().image_processor(
            images=img, return_tensors="pt"
        )["pixel_values"]
        px = px.to(_device, dtype=_dtype)

        # Create prompt
        question = "What is happening in this frame?"
        messages = [{"role": "user", "content": f"<image>\n{question}"}]
        rendered = _tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        pre, post = rendered.split("<image>", 1)

        pre_ids = _tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids.to(_device)
        post_ids = _tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids.to(_device)

        img_tok = torch.tensor([[-200]], dtype=pre_ids.dtype, device=_device)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1)
        attn = torch.ones_like(input_ids, device=_device)

        with torch.no_grad():
            out = _model.generate(
                inputs=input_ids,
                attention_mask=attn,
                images=px,
                max_new_tokens=64,
                do_sample=False,
            )
        
        result = _tokenizer.decode(out[0], skip_special_tokens=True)
        
        # Extract response
        if "What is happening in this frame?" in result:
            result = result.split("What is happening in this frame?")[-1].strip()
        
        return result
        
    except Exception as e:
        return f"Error analyzing frame: {e}"
```

### 2.2 Create Python Dependencies
```bash
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
transformers>=4.30.0
maturin>=1.0.0
```

## 🦀 Step 3: Rust PyO3 Module

### 3.1 Create PyO3 Module
```rust
// src/lib.rs
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::SystemTime;

#[pyclass]
struct FFmpegCapture {
    child: std::process::Child,
    stdout: std::process::ChildStdout,
    width: usize,
    height: usize,
    frame_size: usize,
    frame_count: usize,
    start_time: SystemTime,
}

#[pymethods]
impl FFmpegCapture {
    #[new]
    #[pyo3(signature = (device=None, width=None, height=None, framerate=None))]
    fn new(
        device: Option<String>,
        width: Option<usize>,
        height: Option<usize>,
        framerate: Option<u32>,
    ) -> PyResult<Self> {
        let device = device.unwrap_or_else(|| "0".to_string());
        let width = width.unwrap_or(640);
        let height = height.unwrap_or(480);
        let framerate = framerate.unwrap_or(5);

        // Spawn ffmpeg
        let mut child = Command::new("ffmpeg")
            .args([
                "-hide_banner", "-loglevel", "error",
                "-f", "avfoundation",
                "-framerate", &framerate.to_string(),
                "-video_size", &format!("{width}x{height}"),
                "-i", &device,
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to spawn ffmpeg: {e}")))?;

        let stdout = child.stdout.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Failed to capture ffmpeg stdout")
        })?;

        let frame_size = width * height * 3;
        let start_time = SystemTime::now();

        Ok(Self {
            child,
            stdout,
            width,
            height,
            frame_size,
            frame_count: 0,
            start_time,
        })
    }

    fn next_frame<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let mut buf = vec![0u8; self.frame_size];
        let mut read_bytes = 0usize;

        while read_bytes < self.frame_size {
            match self.stdout.read(&mut buf[read_bytes..]) {
                Ok(0) => {
                    if read_bytes == 0 {
                        return Ok(None);
                    } else {
                        return Err(pyo3::exceptions::PyEOFError::new_err(
                            "Unexpected EOF in middle of frame",
                        ));
                    }
                }
                Ok(n) => read_bytes += n,
                Err(e) => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Read error: {e}"
                    )))
                }
            }
        }

        self.frame_count += 1;
        Ok(Some(PyBytes::new_bound(py, &buf)))
    }

    fn frame_info(&self) -> PyResult<(f64, usize)> {
        let elapsed = self.start_time.elapsed()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Time error: {e}")))?;
        let timestamp = elapsed.as_secs_f64();
        Ok((timestamp, self.frame_count))
    }

    fn close(&mut self) -> PyResult<()> {
        let _ = self.child.kill();
        let _ = self.child.wait();
        Ok(())
    }
}

#[pymodule]
fn camstream(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<FFmpegCapture>()?;
    Ok(())
}
```

### 3.2 Create Rust Binary
```rust
// src/main_simple.rs
use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn main() {
    println!("🚀 Camera Stream Analyzer");
    println!("=========================");
    
    // Set up signal handling
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    
    ctrlc::set_handler(move || {
        println!("\n🛑 Shutdown requested...");
        r.store(false, Ordering::SeqCst);
    }).expect("Error setting Ctrl+C handler");
    
    // Get current directory
    let current_dir = std::env::current_dir().expect("Failed to get current directory");
    
    // Run Python script
    let mut child = Command::new(".venv/bin/python")
        .arg("rust_camera_cli.py")
        .arg("0")           // device
        .arg("640")         // width
        .arg("480")         // height
        .arg("30")          // framerate
        .arg("unlimited")   // max_frames
        .arg("15")          // frame_skip
        .current_dir(&current_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to execute Python script");
    
    // Stream output
    if let Some(stdout) = child.stdout.take() {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            if !running.load(Ordering::SeqCst) {
                break;
            }
            if let Ok(line) = line {
                println!("{}", line);
            }
        }
    }
    
    // Handle stderr
    if let Some(stderr) = child.stderr.take() {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(line) = line {
                eprintln!("⚠️ {}", line);
            }
        }
    }
    
    // Wait for completion
    let status = if running.load(Ordering::SeqCst) {
        child.wait().expect("Failed to wait for Python process")
    } else {
        let _ = child.kill();
        println!("\n🛑 Process terminated gracefully");
        return;
    };
    
    if status.success() {
        println!("\n✅ Analysis completed successfully!");
    } else {
        println!("\n❌ Process exited with error code: {}", status);
    }
}
```

## 🔗 Step 4: Python-Rust Bridge

### 4.1 Create Python CLI
```python
# rust_camera_cli.py
import sys
import time
import signal
from camstream import FFmpegCapture
from fastvlm_analyzer import analyze_frame

# Global flag for shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print("\n🛑 Shutdown requested...")
    shutdown_requested = True

# Set up signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main function called by Rust"""
    print("🤖 Python FastVLM module initialized by Rust")
    
    # Get arguments from Rust
    device = sys.argv[1] if len(sys.argv) > 1 else "0"
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    height = int(sys.argv[3]) if len(sys.argv) > 3 else 480
    framerate = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    max_frames = None if len(sys.argv) > 5 and sys.argv[5] == "unlimited" else int(sys.argv[5]) if len(sys.argv) > 5 else None
    frame_skip = int(sys.argv[6]) if len(sys.argv) > 6 else 15
    
    print(f"📹 Camera settings: {width}x{height} @ {framerate} FPS")
    print(f"📊 Processing every {frame_skip} frames")
    
    try:
        cap = FFmpegCapture(device=device, width=width, height=height, framerate=framerate)
        
        with cap:
            frame_count = 0
            total_frames_read = 0
            start_time = time.time()
            
            while (max_frames is None or frame_count < max_frames) and not shutdown_requested:
                frame = cap.next_frame()
                if frame is None:
                    print("📹 No more frames available")
                    break
                
                if shutdown_requested:
                    break
                
                total_frames_read += 1
                
                # Skip frames for efficiency
                if total_frames_read % frame_skip != 0:
                    continue
                
                # Analyze frame
                analysis = analyze_frame(frame, width, height)
                
                # Get timing info
                timestamp, _ = cap.frame_info()
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Output result
                print(f"[Frame {frame_count}] [t={timestamp:.1f}s] [fps={fps:.1f}] {analysis}")
                
                frame_count += 1
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\n📊 Summary: Processed {frame_count} frames in {total_time:.1f}s (avg {avg_fps:.1f} fps)")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## 🔨 Step 5: Build and Test

### 5.1 Build PyO3 Module
```bash
# Install maturin
pip install maturin

# Build the module
maturin develop
```

### 5.2 Build Rust Binary
```bash
cargo build --bin camstream
```

### 5.3 Test the System
```bash
# Test Python components
python -c "import fastvlm_analyzer; print('FastVLM OK')"
python -c "import camstream; print('PyO3 OK')"

# Test camera
python test_camera.py

# Run full system
./target/debug/camstream
```

## 📦 Step 6: Project Polish

### 6.1 Add Configuration
```python
# config.py
CAMERA_CONFIG = {
    "device": "0",
    "width": 640,
    "height": 480,
    "framerate": 30,
}

PROCESSING_CONFIG = {
    "max_frames": None,
    "question": "What is happening in this frame?",
    "max_new_tokens": 64,
}
```

### 6.2 Add Setup Script
```bash
#!/bin/bash
# setup.sh
echo "🎥 Setting up Camera Analyzer..."

# Install system dependencies
brew install ffmpeg

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Build PyO3 module
maturin develop

# Build Rust binary
cargo build --bin camstream

echo "✅ Setup complete!"
```

### 6.3 Add Documentation
```markdown
# README.md
# Camera Stream Analyzer

A Rust-driven real-time camera analyzer using FastVLM AI.

## Quick Start
```bash
./setup.sh
source .venv/bin/activate
./target/debug/camstream
```
```

## 🎯 Key Learning Points

### 1. **PyO3 Integration**
- Rust provides performance-critical camera capture
- Python handles AI model loading and inference
- PyO3 creates seamless bindings between them

### 2. **Process Architecture**
- Rust binary orchestrates the entire pipeline
- Python subprocess handles AI analysis
- Real-time communication via stdout/stderr

### 3. **Error Handling**
- Graceful shutdown with signal handling
- Proper error propagation between languages
- Robust camera and model initialization

### 4. **Performance Optimization**
- Frame skipping for efficiency
- Model caching to avoid reloading
- Async processing where possible

## 🚀 Next Steps

1. **Add more AI models** (CLIP, BLIP, etc.)
2. **Implement video recording** with analysis
3. **Add web interface** for remote monitoring
4. **Create mobile app** companion
5. **Add cloud deployment** options

## 📚 Resources

- [PyO3 Documentation](https://pyo3.rs/)
- [FastVLM Model](https://huggingface.co/apple/FastVLM-0.5B)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [Rust Book](https://doc.rust-lang.org/book/)

---
*This guide shows how to build a complete Rust-Python AI camera analyzer from scratch!*
