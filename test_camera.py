#!/usr/bin/env python3
"""
Simple test script to verify camera capture works.
"""

from camstream import FFmpegCapture, list_camera_devices
import time

def main():
    print("🧪 Testing camera capture...")
    
    # List available devices
    print("📷 Available devices:")
    devices = list_camera_devices()
    print(f"  Devices: {devices}")
    
    if not devices:
        print("⚠️ No devices found, trying default device 0")
        device = "0"
    else:
        device = devices[0]
    
    print(f"📹 Using device: {device}")
    
    try:
        # Try to create capture
        cap = FFmpegCapture(device=device, width=320, height=240, framerate=1)
        print("✅ FFmpegCapture created successfully")
        
        # Try to read one frame
        with cap:
            print("🔄 Attempting to read frame...")
            frame = cap.next_frame()
            
            if frame is not None:
                print(f"✅ Successfully read frame: {len(frame)} bytes")
                print(f"📊 Frame info: {cap.frame_info()}")
            else:
                print("❌ No frame received")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure camera permissions are granted")
        print("2. Check if camera is being used by another application")
        print("3. Try different device indices")

if __name__ == "__main__":
    main()
