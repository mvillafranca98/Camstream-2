#!/usr/bin/env python3
"""
Test script specifically for Mac built-in camera with correct framerate settings.
"""

from camstream import FFmpegCapture, list_camera_devices, validate_frame
import time

def test_camera_with_framerates():
    """Test camera with different framerates that work on Mac"""
    print("🧪 Testing Mac built-in camera with different framerates...")
    
    # Test framerates that typically work on Mac
    test_framerates = [15, 30]
    width, height = 640, 480
    
    for framerate in test_framerates:
        print(f"\n📹 Testing framerate: {framerate} FPS")
        
        try:
            cap = FFmpegCapture(device="0", width=width, height=height, framerate=framerate)
            print(f"✅ FFmpegCapture created successfully for {framerate} FPS")
            
            with cap:
                print("🔄 Attempting to read frames...")
                frames_read = 0
                start_time = time.time()
                
                # Try to read a few frames
                for i in range(5):
                    frame = cap.next_frame()
                    if frame is None:
                        print(f"❌ No frame {i} received")
                        break
                    
                    if validate_frame(frame, width, height):
                        frames_read += 1
                        print(f"✅ Frame {i}: {len(frame)} bytes")
                    else:
                        print(f"⚠️ Frame {i}: Invalid size")
                
                elapsed = time.time() - start_time
                actual_fps = frames_read / elapsed if elapsed > 0 else 0
                print(f"📊 Read {frames_read} frames in {elapsed:.2f}s (actual FPS: {actual_fps:.1f})")
                
                if frames_read > 0:
                    print(f"🎉 SUCCESS: {framerate} FPS works!")
                    return framerate
                else:
                    print(f"❌ FAILED: {framerate} FPS didn't work")
                    
        except Exception as e:
            print(f"❌ Error with {framerate} FPS: {e}")
            continue
    
    print("\n⚠️ None of the tested framerates worked")
    return None

def main():
    print("🔍 Mac Camera Framerate Test")
    print("=" * 40)
    
    # List devices
    print("📷 Available devices:")
    devices = list_camera_devices()
    print(f"  Devices: {devices}")
    
    if not devices:
        print("⚠️ No devices found - this is normal if camera permissions aren't granted")
        print("💡 To grant permissions:")
        print("   1. Go to System Preferences > Security & Privacy > Camera")
        print("   2. Enable camera access for Terminal")
        print("   3. Close any other apps using the camera (FaceTime, Zoom, etc.)")
    
    # Test framerates
    working_framerate = test_camera_with_framerates()
    
    if working_framerate:
        print(f"\n🎉 Recommended settings:")
        print(f"   Framerate: {working_framerate} FPS")
        print(f"   Frame skip: {working_framerate // 2} (for 2 FPS processing)")
        print(f"\n💡 Update your config.py:")
        print(f"   CAMERA_CONFIG['framerate'] = {working_framerate}")
        print(f"   PROCESSING_CONFIG['frame_skip'] = {working_framerate // 2}")
    else:
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure camera permissions are granted")
        print("2. Close other apps using the camera")
        print("3. Try restarting Terminal")
        print("4. Check if camera is working in other apps")

if __name__ == "__main__":
    main()

