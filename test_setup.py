#!/usr/bin/env python3
"""
Test script to verify the camera stream setup and dependencies.
"""

import sys
import subprocess

def check_ffmpeg():
    """Check if ffmpeg is installed and accessible"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg is installed and accessible")
            return True
        else:
            print("❌ FFmpeg is not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ FFmpeg is not installed or not in PATH")
        return False

def check_python_packages():
    """Check if required Python packages are installed"""
    required_packages = [
        'torch', 'PIL', 'transformers', 'pyo3'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'pyo3':
                # PyO3 is used via the compiled module
                import camstream
            else:
                __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is not installed")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def test_camera_detection():
    """Test camera device detection"""
    try:
        from camstream import list_camera_devices
        devices = list_camera_devices()
        print(f"✅ Found {len(devices)} camera devices: {devices}")
        return True
    except Exception as e:
        print(f"❌ Camera detection failed: {e}")
        return False

def test_module_import():
    """Test if the camstream module can be imported"""
    try:
        from camstream import FFmpegCapture, validate_frame, frame_to_pil_image
        print("✅ camstream module imports successfully")
        return True
    except Exception as e:
        print(f"❌ camstream module import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 Testing Camera Stream Setup")
    print("=" * 40)
    
    tests = [
        ("FFmpeg Installation", check_ffmpeg),
        ("Python Packages", lambda: check_python_packages()[0]),
        ("Camera Detection", test_camera_detection),
        ("Module Import", test_module_import),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 40)
    print("📊 Test Results Summary:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("You can now run: python live_fastvlm_enhanced.py")
    else:
        print("\n⚠️ Some tests failed. Please check the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
