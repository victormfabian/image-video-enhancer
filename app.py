import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import tempfile
import io
import os

st.set_page_config(page_title="🔍 Image & Video Enhancer", layout="wide")
st.title("🔍 High-Quality Image & Video Enhancer")

# Load model with local-only check
@st.cache_resource
def load_model(scale):
    model_name = f"RealESRGAN_x{scale}plus.pth"
    model_dir = os.path.join("models")  # Use local models directory
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)

    if not os.path.exists(model_path):
        st.error(f"Model file not found at:\n{model_path}\n\nPlease download it manually from:\nhttps://github.com/xinntao/Real-ESRGAN/releases and place it in the 'models' folder.")
        st.stop()

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True if torch.cuda.is_available() else False
    )
    return upsampler

mode = st.radio("Select Mode", ["Image", "Video"], horizontal=True)
scale = st.select_slider("Enhancement Scale", options=[2, 4], value=4)

if mode == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        st.image(image, caption="Original", use_container_width=True)

        with st.spinner("Enhancing image..."):
            upsampler = load_model(scale)
            output, _ = upsampler.enhance(img_np, outscale=scale)
            result_img = Image.fromarray(output)

        st.image(result_img, caption=f"Enhanced x{scale}", use_container_width=True)

        img_bytes = io.BytesIO()
        result_img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        st.download_button(
            label="Download Enhanced Image",
            data=img_bytes,
            file_name=f"enhanced_x{scale}.png",
            mime="image/png"
        )

else:
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(video_path)
        st.warning("⚠️ Video enhancement preview is limited. Full processing may take time.")

        if st.button("Process Video"):
            with st.spinner("Enhancing video frame by frame (this may take a while)..."):
                upsampler = load_model(scale)
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                output_path = video_path.replace(".", f"_enhanced_x{scale}.")
                out = cv2.VideoWriter(output_path, fourcc, fps, (width * scale, height * scale))

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    enhanced_frame, _ = upsampler.enhance(frame, outscale=scale)
                    out.write(enhanced_frame)

                cap.release()
                out.release()

            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download Enhanced Video",
                    data=f,
                    file_name=f"enhanced_x{scale}.mp4",
                    mime="video/mp4"
                )
