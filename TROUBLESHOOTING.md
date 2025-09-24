# 🔧 Camstream-2 Troubleshooting Guide

## 🚨 Quick Fixes

### 1. Camera Not Working
```bash
# Check camera permissions
# System Preferences > Security & Privacy > Camera > Terminal

# List available cameras
ffmpeg -f avfoundation -list_devices true -i ""

# Try different device index
./target/debug/camstream  # Uses device "0"
```

### 2. "Command not found" Errors
```bash
# Activate virtual environment
source .venv/bin/activate

# Check if binary exists
ls -la target/debug/camstream

# Rebuild if missing
cargo build --bin camstream
```

### 3. Python Module Errors
```bash
# Reinstall PyO3 module
maturin develop

# Reinstall Python dependencies
pip install --upgrade -r requirements.txt

# Check Python path
which python
```

### 4. Build Failures
```bash
# Clean everything
cargo clean
rm -rf target/

# Rebuild from scratch
maturin develop
cargo build --bin camstream
```

## 🔍 Diagnostic Commands

### Check System Status
```bash
# Check FFmpeg
ffmpeg -version

# Check Rust
cargo --version

# Check Python
python --version
which python

# Check virtual environment
source .venv/bin/activate
which python
```

### Test Individual Components
```bash
# Test camera only
python test_mac_camera.py

# Test setup
python test_setup.py

# Test FastVLM only
python -c "import fastvlm_analyzer; print('FastVLM OK')"
```

## 🐛 Common Error Messages

### "Device not configured"
- **Cause**: Git authentication issue
- **Fix**: Use Personal Access Token or SSH

### "Camera permission denied"
- **Cause**: macOS security settings
- **Fix**: Grant camera access to Terminal

### "Module not found"
- **Cause**: Virtual environment not activated
- **Fix**: Run `source .venv/bin/activate`

### "Failed to spawn ffmpeg"
- **Cause**: FFmpeg not installed or not in PATH
- **Fix**: `brew install ffmpeg`

### "No camera devices found"
- **Cause**: Camera not connected or permissions denied
- **Fix**: Check connections and permissions

## 📋 Pre-Flight Checklist

Before running Camstream-2, verify:

- [ ] FFmpeg installed (`ffmpeg -version`)
- [ ] Rust installed (`cargo --version`)
- [ ] Python 3.8+ installed (`python --version`)
- [ ] Virtual environment created (`.venv` folder exists)
- [ ] Dependencies installed (`pip list | grep torch`)
- [ ] PyO3 module built (`maturin develop`)
- [ ] Rust binary built (`ls target/debug/camstream`)
- [ ] Camera permissions granted
- [ ] Camera connected and working

## 🆘 Emergency Reset

If nothing works, start fresh:

```bash
# Remove virtual environment
rm -rf .venv

# Remove build artifacts
cargo clean
rm -rf target/

# Recreate everything
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
maturin develop
cargo build --bin camstream
```

## 📞 Getting Help

1. **Check this guide first**
2. **Run diagnostic commands**
3. **Check GitHub issues**: https://github.com/mvillafranca98/Camstream-2/issues
4. **Create new issue** with:
   - Error message
   - System info (`uname -a`)
   - Python version (`python --version`)
   - Steps to reproduce

---
*Keep this guide handy for quick reference!*
