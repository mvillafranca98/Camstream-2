"""
Python module for FastVLM analysis that can be called from Rust via PyO3.
"""

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

# Global variables to cache the model and tokenizer
_model = None
_tokenizer = None
_device = None
_dtype = None

def initialize_model():
    """Initialize the FastVLM model and tokenizer (cached)"""
    global _model, _tokenizer, _device, _dtype
    
    if _model is not None:
        return  # Already initialized
    
    MODEL_ID = "apple/FastVLM-0.5B"
    IMAGE_TOKEN_INDEX = -200
    
    # Detect device
    if torch.backends.mps.is_available():
        _device = torch.device("mps")
        _dtype = torch.float16
        print("✅ Using Apple MPS backend")
    elif torch.cuda.is_available():
        _device = torch.device("cuda")
        _dtype = torch.float16
        print("✅ Using CUDA GPU")
    else:
        _device = torch.device("cpu")
        _dtype = torch.float32
        print("⚠️ Running on CPU only")
    
    # Load model and tokenizer
    print(f"🤖 Loading model: {MODEL_ID}")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=_dtype,
        device_map=None,
        trust_remote_code=True,
    )
    _model.to(_device)
    
    print("✅ FastVLM model loaded successfully")

def analyze_frame(frame_bytes, width, height):
    """
    Analyze a frame using FastVLM.
    
    Args:
        frame_bytes: Raw RGB frame data as bytes
        width: Frame width
        height: Frame height
    
    Returns:
        String description of the frame
    """
    global _model, _tokenizer, _device, _dtype
    
    # Initialize model if not already done
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

        # Chat-style prompt
        question = "What is happening in this frame?"
        messages = [{"role": "user", "content": f"<image>\n{question}"}]
        rendered = _tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        pre, post = rendered.split("<image>", 1)

        pre_ids = _tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids.to(_device)
        post_ids = _tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids.to(_device)

        img_tok = torch.tensor(
            [[-200]],  # IMAGE_TOKEN_INDEX
            dtype=pre_ids.dtype, device=_device
        )
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
        
        # Extract just the response part (remove the prompt)
        if "What is happening in this frame?" in result:
            result = result.split("What is happening in this frame?")[-1].strip()
        
        return result
        
    except Exception as e:
        return f"Error analyzing frame: {e}"

def cleanup():
    """Clean up resources"""
    global _model, _tokenizer
    _model = None
    _tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

