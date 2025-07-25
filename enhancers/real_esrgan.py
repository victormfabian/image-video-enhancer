# enhancers/real_esrgan.py
import os
import torch
import numpy as np
from PIL import Image
import cv2
from realesrgan import RealESRGAN


def load_model(scale=4, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = RealESRGAN(device, scale=scale)
    model.load_weights(f'models/RealESRGAN_x{scale}.pth')
    return model


def enhance_image(image: Image.Image, model):
    return model.predict(image)


def enhance_video(video_path, output_path, model):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        enhanced = model.predict(pil_img)

        enhanced_bgr = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)

        if out is None:
            h, w = enhanced_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        out.write(enhanced_bgr)

    cap.release()
    out.release()
