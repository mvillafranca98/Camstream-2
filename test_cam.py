from camstream import FFmpegCapture
from PIL import Image

WIDTH, HEIGHT = 640, 480
cap = FFmpegCapture(device="0", width=WIDTH, height=HEIGHT, framerate=30)

count = 0
with cap:
    while count < 5:
        frame = cap.next_frame()
        if frame is None:
            print("No more frames (None).")
            break
        print(f"Got frame {count}, bytes={len(frame)}")
        img = Image.frombytes("RGB", (WIDTH, HEIGHT), frame)
        img.save(f"frame_{count}.jpg", quality=85)
        count += 1
