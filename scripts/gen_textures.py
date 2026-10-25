import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

IMAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")

def height_to_normal (height: np.ndarray, strength: float) -> np.ndarray:
  gx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5
  gy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5
  n = np.dstack([-strength * gx, strength * gy, np.ones_like(height)])
  n /= np.linalg.norm(n, axis=2, keepdims=True)
  return ((n * 0.5 + 0.5) * 255).astype(np.uint8)

def normal_from_image (src: str, dst: str, strength: float, blur: float) -> None:
  img = Image.open(os.path.join(IMAGES, src)).convert("L").filter(ImageFilter.GaussianBlur(blur))
  height = np.asarray(img, dtype=np.float32) / 255.0
  Image.fromarray(height_to_normal(height, strength), "RGB").save(os.path.join(IMAGES, dst))

def make_slide (dst: str, size: int = 512) -> None:
  img = Image.new("RGB", (size, size), (20, 20, 40))
  d = ImageDraw.Draw(img)
  colors = [(230, 40, 40), (240, 160, 30), (240, 230, 50), (60, 190, 70), (50, 120, 230), (150, 70, 200)]
  w = size // len(colors)
  for i, c in enumerate(colors):
    d.rectangle([i * w, 0, (i + 1) * w, size // 3], fill=c)
  d.rectangle([size // 12, size * 2 // 3, size // 3, size * 11 // 12], fill=(255, 255, 255))
  d.polygon([(size * 2 // 3, size * 11 // 12), (size * 11 // 12, size * 11 // 12), (size * 11 // 12, size * 2 // 3)], fill=(255, 90, 160))
  d.text((size // 4, size * 5 // 12), "INF1761", fill=(255, 255, 255), font_size=size // 6)
  d.text((size // 3, size * 7 // 12), "PUC-Rio", fill=(255, 255, 255), font_size=size // 10)
  d.rectangle([0, 0, size - 1, size - 1], outline=(255, 255, 255), width=6)
  img.save(os.path.join(IMAGES, dst))

if __name__ == "__main__":
  normal_from_image("wood.png", "wood_normal.png", strength=20.0, blur=1.0)
  normal_from_image("giftwrap.png", "giftwrap_normal.png", strength=12.0, blur=0.8)
  make_slide("slide.png")
  print("normal maps e slide gerados em", os.path.normpath(IMAGES))
