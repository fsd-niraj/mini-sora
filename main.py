import torch
from diffusers import StableDiffusionXLPipeline
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as T
import os
import numpy as np
from PIL import Image
import subprocess

# -------------------------------------------------------
# Step 1 — Text → Image (using SD-Turbo for speed/safety)
# -------------------------------------------------------
def generate_base_image(prompt):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("mps")

    img = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
    return img


# -------------------------------------------------------
# Step 2 — Tiny motion function (risk-free "fake motion")
# -------------------------------------------------------
def add_tiny_motion(img, t):
    """
    Adds small warps + noise changes to simulate motion.
    Risk-free because we do NOT preserve any identity.
    """
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0

    # Tiny x-shift
    shift = int(2 * np.sin(t * 0.3))
    arr = np.roll(arr, shift, axis=1)

    # Add a bit of procedural noise
    noise = (np.random.randn(*arr.shape) * 0.01).astype(np.float32)
    arr = np.clip(arr + noise, 0, 1)

    return Image.fromarray((arr * 255).astype(np.uint8))


# -------------------------------------------------------
# Step 3 — Generate video frames
# -------------------------------------------------------
def generate_frames(prompt, n_frames=100, out_folder="output/frames"):
    os.makedirs(out_folder, exist_ok=True)

    # Base frame
    base = generate_base_image(prompt)

    for i in range(n_frames):
        frame = add_tiny_motion(base, i)
        frame.save(f"{out_folder}/frame_{i:04d}.png")

    print(f"[✓] Generated {n_frames} frames.")


# -------------------------------------------------------
# Step 4 — Stitch frames → video
# -------------------------------------------------------
def make_video(out_path="output/out.mp4"):
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", "10",
        "-i", "output/frames/frame_%04d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_path
    ]
    subprocess.run(cmd)
    print("[✓] Video created!", out_path)
    
def generate_video_from_prompt(prompt, duration=2, fps=12):
    generate_frames(prompt)
    make_video()
    return "output/out.mp4"


# if __name__ == "__main__":
    # prompt = "a minion playing guitar on the beach, cinematic lighting"
    # generate_frames(prompt)
    # make_video()
