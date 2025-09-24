import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from camstream import FFmpegCapture, list_camera_devices, validate_frame
import time
import signal
import sys

MODEL_ID = "apple/FastVLM-0.5B"
IMAGE_TOKEN_INDEX = -200  # from model card

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print("\n🛑 Shutdown requested...")
    shutdown_requested = True

# Set up signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ---------------------------
# Device + dtype detection
# ---------------------------
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

# ---------------------------
# Camera device detection
# ---------------------------
print("📷 Available camera devices:")
try:
    devices = list_camera_devices()
    for i, device_id in enumerate(devices):
        print(f"  {i}: Device {device_id}")
    if not devices:
        print("  No devices found, using default device 0")
        selected_device = "0"
    else:
        selected_device = devices[0]  # Use first available device
except Exception as e:
    print(f"  Could not list devices: {e}")
    selected_device = "0"

print(f"📹 Using camera device: {selected_device}")

# ---------------------------
# Load model & tokenizer
# ---------------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=dtype,              # ✅ no more deprecation warning
    device_map=None,          # we’ll move it manually
    trust_remote_code=True,
)
model.to(device)

# ---------------------------
# Camera capture configuration
# ---------------------------
WIDTH, HEIGHT = 640, 480
FRAMERATE = 30  # Frames per second (Mac built-in camera requires 15 or 30)
MAX_FRAMES = 100  # Maximum frames to process (None for unlimited)
FRAME_SKIP = 15  # Process every Nth frame (30fps / 15 = 2fps processing)

print(f"🎥 Camera settings: {WIDTH}x{HEIGHT} @ {FRAMERATE} FPS (processing every {FRAME_SKIP} frames)")

def describe(img: Image.Image, question="What is happening in this frame?"):
    # Preprocess image
    px = model.get_vision_tower().image_processor(images=img, return_tensors="pt")["pixel_values"]
    px = px.to(device, dtype=dtype)

    # Chat-style prompt with <image> placeholder
    messages = [{"role": "user", "content": f"<image>\n{question}"}]
    rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    pre, post = rendered.split("<image>", 1)

    pre_ids  = tok(pre,  return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype, device=device)
    input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1)
    attn = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        out = model.generate(
            inputs=input_ids,
            attention_mask=attn,
            images=px,
            max_new_tokens=64,
            do_sample=False,
        )
    return tok.decode(out[0], skip_special_tokens=True)

# ---------------------------
# Main processing loop
# ---------------------------
print("🚀 Starting camera capture and analysis...")
print("Press Ctrl+C to stop gracefully")

try:
    cap = FFmpegCapture(device=selected_device, width=WIDTH, height=HEIGHT, framerate=FRAMERATE)
    
    with cap:
        frame_count = 0
        total_frames_read = 0
        start_time = time.time()
        
        while not shutdown_requested:
            if MAX_FRAMES is not None and frame_count >= MAX_FRAMES:
                print(f"✅ Processed {MAX_FRAMES} frames as requested")
                break
                
            frame = cap.next_frame()
            if frame is None:
                print("📹 No more frames available")
                break
            
            total_frames_read += 1
            
            # Skip frames to reduce processing load
            if total_frames_read % FRAME_SKIP != 0:
                continue
            
            # Validate frame
            if not validate_frame(frame, WIDTH, HEIGHT):
                print(f"⚠️ Invalid frame {frame_count}, skipping...")
                continue
            
            # Convert to PIL Image
            try:
                img = Image.frombytes("RGB", (WIDTH, HEIGHT), frame)
                
                # Get frame timing info
                timestamp, cap_frame_count = cap.frame_info()
                
                # Process with FastVLM
                answer = describe(img)
                
                # Display results
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"[Frame {frame_count}] [t={timestamp:.1f}s] [fps={fps:.1f}] {answer}")
                
                frame_count += 1
                
            except Exception as e:
                print(f"❌ Error processing frame {frame_count}: {e}")
                continue

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
finally:
    total_time = time.time() - start_time if 'start_time' in locals() else 0
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\n📊 Summary: Processed {frame_count} frames in {total_time:.1f}s (avg {avg_fps:.1f} fps)")
    print("👋 Goodbye!")

