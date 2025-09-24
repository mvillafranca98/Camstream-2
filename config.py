"""
Configuration settings for the camera stream and FastVLM processing.
"""

# Camera settings
CAMERA_CONFIG = {
    "device": None,  # None = auto-detect, or specify device index as string
    "width": 640,
    "height": 480,
    "framerate": 30,  # Frames per second (Mac built-in camera requires 15 or 30)
    "pixel_format": "uyvy422",  # Input pixel format from camera
}

# Processing settings
PROCESSING_CONFIG = {
    "max_frames": 100,  # None for unlimited
    "question": "What is happening in this frame?",
    "max_new_tokens": 64,
    "do_sample": False,
    "frame_skip": 15,  # Process every Nth frame (30fps / 15 = 2fps processing)
}

# Model settings
MODEL_CONFIG = {
    "model_id": "apple/FastVLM-0.5B",
    "image_token_index": -200,
    "trust_remote_code": True,
}

# Output settings
OUTPUT_CONFIG = {
    "show_timestamps": True,
    "show_fps": True,
    "show_frame_info": True,
    "save_frames": False,  # Set to True to save frames as images
    "output_dir": "frames",  # Directory to save frames if save_frames is True
}

# Advanced settings
ADVANCED_CONFIG = {
    "frame_validation": True,
    "graceful_shutdown": True,
    "error_recovery": True,
    "verbose_logging": False,
}
