#!/usr/bin/env python3
"""
Enhanced Live FastVLM Camera Stream Analyzer

This script captures video from your camera using ffmpeg, processes frames one by one,
and sends them to FastVLM for real-time scene analysis.

Usage:
    python live_fastvlm_enhanced.py

Press Ctrl+C to stop gracefully.
"""

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from camstream import FFmpegCapture, list_camera_devices, validate_frame
import time
import signal
import sys
import os
from config import (
    CAMERA_CONFIG, PROCESSING_CONFIG, MODEL_CONFIG, 
    OUTPUT_CONFIG, ADVANCED_CONFIG
)

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print("\n🛑 Shutdown requested...")
    shutdown_requested = True

def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown"""
    if ADVANCED_CONFIG["graceful_shutdown"]:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

def detect_device():
    """Detect and select the best available camera device"""
    print("📷 Available camera devices:")
    try:
        devices = list_camera_devices()
        for i, device_id in enumerate(devices):
            print(f"  {i}: Device {device_id}")
        
        if not devices:
            print("  No devices found, using default device 0")
            print("  Note: This is normal if camera permissions haven't been granted yet")
            return "0"
        
        # Use configured device or first available
        if CAMERA_CONFIG["device"] is not None:
            if CAMERA_CONFIG["device"] in devices:
                return CAMERA_CONFIG["device"]
            else:
                print(f"⚠️ Configured device {CAMERA_CONFIG['device']} not found, using first available")
        
        return devices[0]
    except Exception as e:
        print(f"  Could not list devices: {e}")
        print("  Note: This is normal if camera permissions haven't been granted yet")
        return "0"

def setup_output_directory():
    """Create output directory if saving frames is enabled"""
    if OUTPUT_CONFIG["save_frames"]:
        os.makedirs(OUTPUT_CONFIG["output_dir"], exist_ok=True)
        print(f"📁 Frames will be saved to: {OUTPUT_CONFIG['output_dir']}")

def detect_compute_device():
    """Detect the best available compute device (MPS, CUDA, or CPU)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
        print("✅ Using Apple MPS backend")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
        print("✅ Using CUDA GPU")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("⚠️ Running on CPU only (this will be slower)")
    
    return device, dtype

def load_model_and_tokenizer(device, dtype):
    """Load the FastVLM model and tokenizer"""
    print(f"🤖 Loading model: {MODEL_CONFIG['model_id']}")
    
    tok = AutoTokenizer.from_pretrained(
        MODEL_CONFIG["model_id"], 
        trust_remote_code=MODEL_CONFIG["trust_remote_code"]
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["model_id"],
        dtype=dtype,
        device_map=None,  # We'll move it manually
        trust_remote_code=MODEL_CONFIG["trust_remote_code"],
    )
    model.to(device)
    
    return model, tok

def describe_frame(model, tok, device, dtype, img, question=None):
    """Analyze a frame using FastVLM"""
    if question is None:
        question = PROCESSING_CONFIG["question"]
    
    try:
        # Preprocess image
        px = model.get_vision_tower().image_processor(
            images=img, return_tensors="pt"
        )["pixel_values"]
        px = px.to(device, dtype=dtype)

        # Chat-style prompt with <image> placeholder
        messages = [{"role": "user", "content": f"<image>\n{question}"}]
        rendered = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        pre, post = rendered.split("<image>", 1)

        pre_ids = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        img_tok = torch.tensor(
            [[MODEL_CONFIG["image_token_index"]]], 
            dtype=pre_ids.dtype, device=device
        )
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1)
        attn = torch.ones_like(input_ids, device=device)

        with torch.no_grad():
            out = model.generate(
                inputs=input_ids,
                attention_mask=attn,
                images=px,
                max_new_tokens=PROCESSING_CONFIG["max_new_tokens"],
                do_sample=PROCESSING_CONFIG["do_sample"],
            )
        return tok.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error analyzing frame: {e}"

def save_frame_if_enabled(img, frame_count):
    """Save frame to disk if enabled in config"""
    if OUTPUT_CONFIG["save_frames"]:
        filename = f"frame_{frame_count:06d}.jpg"
        filepath = os.path.join(OUTPUT_CONFIG["output_dir"], filename)
        img.save(filepath)

def format_output(frame_count, timestamp, fps, answer):
    """Format the output line with configured information"""
    parts = [f"[Frame {frame_count}]"]
    
    if OUTPUT_CONFIG["show_timestamps"]:
        parts.append(f"[t={timestamp:.1f}s]")
    
    if OUTPUT_CONFIG["show_fps"]:
        parts.append(f"[fps={fps:.1f}]")
    
    parts.append(answer)
    return " ".join(parts)

def main():
    """Main processing loop"""
    print("🚀 Enhanced Live FastVLM Camera Stream Analyzer")
    print("=" * 50)
    
    # Setup
    setup_signal_handlers()
    setup_output_directory()
    
    # Device detection
    device, dtype = detect_compute_device()
    selected_camera = detect_device()
    
    print(f"📹 Using camera device: {selected_camera}")
    print(f"🎥 Camera settings: {CAMERA_CONFIG['width']}x{CAMERA_CONFIG['height']} @ {CAMERA_CONFIG['framerate']} FPS")
    print("Press Ctrl+C to stop gracefully")
    print()
    
    # Load model
    model, tok = load_model_and_tokenizer(device, dtype)
    
    # Main processing loop
    try:
        cap = FFmpegCapture(
            device=selected_camera,
            width=CAMERA_CONFIG["width"],
            height=CAMERA_CONFIG["height"],
            framerate=CAMERA_CONFIG["framerate"],
            pix_in=CAMERA_CONFIG["pixel_format"]
        )
        
        with cap:
            frame_count = 0
            total_frames_read = 0
            start_time = time.time()
            frame_skip = PROCESSING_CONFIG.get("frame_skip", 1)
            
            print(f"📊 Processing every {frame_skip} frame(s) (capture: {CAMERA_CONFIG['framerate']}fps, analysis: {CAMERA_CONFIG['framerate']//frame_skip}fps)")
            
            while not shutdown_requested:
                # Check frame limit
                if (PROCESSING_CONFIG["max_frames"] is not None and 
                    frame_count >= PROCESSING_CONFIG["max_frames"]):
                    print(f"✅ Processed {PROCESSING_CONFIG['max_frames']} frames as requested")
                    break
                
                # Read frame
                frame = cap.next_frame()
                if frame is None:
                    print("📹 No more frames available")
                    break
                
                total_frames_read += 1
                
                # Skip frames if configured
                if total_frames_read % frame_skip != 0:
                    continue
                
                # Validate frame
                if (ADVANCED_CONFIG["frame_validation"] and 
                    not validate_frame(frame, CAMERA_CONFIG["width"], CAMERA_CONFIG["height"])):
                    print(f"⚠️ Invalid frame {frame_count}, skipping...")
                    continue
                
                # Process frame
                try:
                    img = Image.frombytes(
                        "RGB", 
                        (CAMERA_CONFIG["width"], CAMERA_CONFIG["height"]), 
                        frame
                    )
                    
                    # Save frame if enabled
                    save_frame_if_enabled(img, frame_count)
                    
                    # Get timing info
                    timestamp, cap_frame_count = cap.frame_info()
                    
                    # Analyze with FastVLM
                    answer = describe_frame(model, tok, device, dtype, img)
                    
                    # Display results
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    output = format_output(frame_count, timestamp, fps, answer)
                    print(output)
                    
                    frame_count += 1
                    
                except Exception as e:
                    if ADVANCED_CONFIG["error_recovery"]:
                        print(f"❌ Error processing frame {frame_count}: {e}")
                        continue
                    else:
                        raise
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        if not ADVANCED_CONFIG["error_recovery"]:
            sys.exit(1)
    finally:
        # Summary
        if 'start_time' in locals():
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            print(f"\n📊 Summary: Processed {frame_count} frames in {total_time:.1f}s (avg {avg_fps:.1f} fps)")
        
        if OUTPUT_CONFIG["save_frames"]:
            print(f"💾 Frames saved to: {OUTPUT_CONFIG['output_dir']}")
        
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
