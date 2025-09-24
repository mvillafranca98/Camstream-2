#!/usr/bin/env python3
"""
Rust-driven camera analysis using the PyO3 module.
This script is called by Rust and demonstrates the proper PyO3 architecture.
"""

import sys
import time
import signal
from camstream import FFmpegCapture, validate_frame
from fastvlm_analyzer import analyze_frame

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print("\n🛑 Shutdown requested...")
    shutdown_requested = True

# Set up signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main function called by Rust"""
    print("🤖 Python FastVLM module initialized by Rust")
    
    # Configuration (could be passed as arguments from Rust)
    device = sys.argv[1] if len(sys.argv) > 1 else "0"
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    height = int(sys.argv[3]) if len(sys.argv) > 3 else 480
    framerate = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    if len(sys.argv) > 5:
        max_frames_arg = sys.argv[5]
        if max_frames_arg == "unlimited":
            max_frames = None
        else:
            max_frames = int(max_frames_arg)
    else:
        max_frames = None  # None = unlimited
    frame_skip = int(sys.argv[6]) if len(sys.argv) > 6 else 15
    
    print(f"📹 Camera settings: {width}x{height} @ {framerate} FPS")
    print(f"📊 Processing every {frame_skip} frames (analysis rate: {framerate//frame_skip} FPS)")
    if max_frames:
        print(f"🎯 Max frames: {max_frames}")
    else:
        print("🎯 Max frames: Unlimited (Ctrl+C to stop)")
    
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
                
                # Skip frames for processing efficiency
                if total_frames_read % frame_skip != 0:
                    continue
                
                # Validate frame
                if not validate_frame(frame, width, height):
                    print(f"⚠️ Invalid frame {frame_count}, skipping...")
                    continue
                
                # Analyze frame using FastVLM
                analysis = analyze_frame(frame, width, height)
                
                # Get timing info
                timestamp, _ = cap.frame_info()
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Output result (Rust will capture this)
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
