# ===============================================================
# Problem 3 — Edge-preserving smoothing: Bilateral & Nagao–Matsuyama
# ===============================================================
# Requires: numpy, opencv-python (cv2), matplotlib
# (No SciPy needed; we use cv2.filter2D for convolutions.)

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# -----------------------------
# Helper: load image in grayscale [0,1]
# -----------------------------
def load_gray01(path):
    p = Path(path)
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {p}")
    return (img.astype(np.float32) / 255.0), p

# ===============================================================
# 1) Load the bone-scan image (UPDATE THIS PATH)
# ===============================================================
BONE_PATH = "/Users/sana/Desktop/FA25Courses/Comp vision with AI/Projects/Homework2/Bone_scan.tif"  # <- edit to your file
bone, used_path = load_gray01(BONE_PATH)
print(f"[INFO] Bone image loaded from: {used_path} | shape={bone.shape}")

# ===============================================================
# 2) Bilateral filter (edge-preserving)
#    - d: neighborhood diameter (pixels). If negative, computed from sigmaSpace.
#    - sigmaColor: how much intensity differences are smoothed (range kernel).
#    - sigmaSpace: how far pixels influence each other (spatial kernel).
#    - Larger sigmas -> more smoothing but risk haloing / detail loss.
# ===============================================================
def bilateral_filter01(img01, d=9, sigmaColor=40, sigmaSpace=10):
    # OpenCV expects 8-bit or float; we'll keep float in [0,1].
    out = cv2.bilateralFilter(img01.astype(np.float32), d=d,
                              sigmaColor=float(sigmaColor),
                              sigmaSpace=float(sigmaSpace))
    return out

bil_out = bilateral_filter01(bone, d=9, sigmaColor=40, sigmaSpace=10)

# ===============================================================
# 3) Nagao–Matsuyama filter (5×5 window, choose region of min variance)
#    Idea:
#      For each pixel, examine 9 regions inside a 5×5 neighborhood:
#        - center 3×3, plus 8 overlapping 3×3 regions: N,S,E,W, NE,NW,SE,SW
#      Compute each region's mean and variance; pick the mean of the region
#      with the **smallest variance** (assumes it's the most homogeneous).
#    Implementation notes:
#      - We build 9 binary 5×5 masks (normalized), then use cv2.filter2D
#        to compute local means for img and img^2 (to get variance).
#      - Entirely vectorized; no Python loops over pixels.
# ===============================================================
def nagao_matsuyama_5x5(img01):
    img = img01.astype(np.float32)
    img2 = img * img

    # Build 9 normalized 5×5 masks (each covers a 3×3 block inside the 5×5)
    M = np.zeros((9, 5, 5), dtype=np.float32)

    # handy slices for each 3×3 subregion inside 5×5
    blocks = {
        0: (slice(1,4), slice(1,4)),  # center
        1: (slice(0,3), slice(1,4)),  # North
        2: (slice(2,5), slice(1,4)),  # South
        3: (slice(1,4), slice(0,3)),  # West
        4: (slice(1,4), slice(2,5)),  # East
        5: (slice(0,3), slice(0,3)),  # NW
        6: (slice(0,3), slice(2,5)),  # NE
        7: (slice(2,5), slice(0,3)),  # SW
        8: (slice(2,5), slice(2,5)),  # SE
    }
    for i,(rs, cs) in blocks.items():
        M[i, rs, cs] = 1.0 / 9.0  # 3×3 = 9 pixels per region

    # Convolve to get E[X] (means) and E[X^2] for each region (shape H×W per region)
    means = []
    means2 = []
    for i in range(9):
        k = M[i]
        means.append(cv2.filter2D(img,  ddepth=-1, kernel=k, borderType=cv2.BORDER_REFLECT))
        means2.append(cv2.filter2D(img2, ddepth=-1, kernel=k, borderType=cv2.BORDER_REFLECT))
    means  = np.stack(means,  axis=-1)   # H×W×9
    means2 = np.stack(means2, axis=-1)   # H×W×9

    # Var = E[X^2] - (E[X])^2  (per region)
    var = means2 - means * means

    # Pick the region index with minimum variance for each pixel
    idx = np.argmin(var, axis=-1)        # H×W

    # Gather the corresponding mean value
    # Take the per-pixel mean at the chosen region index
    H, W = idx.shape
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    out = means[rows, cols, idx]         # fancy indexing to pick mean per pixel

    return out.astype(np.float32)

nagao_out = nagao_matsuyama_5x5(bone)

# ===============================================================
# 4) Show results
# ===============================================================
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].imshow(bone, cmap='gray');     ax[0].set_title("Original bone scan"); ax[0].axis('off')
ax[1].imshow(bil_out, cmap='gray');  ax[1].set_title("Bilateral");          ax[1].axis('off')
ax[2].imshow(nagao_out, cmap='gray');ax[2].set_title("Nagao–Matsuyama");    ax[2].axis('off')
plt.tight_layout(); plt.show()

# ===============================================================
# 5) Save outputs for your submission
# ===============================================================
outdir = Path("./p3_outputs"); outdir.mkdir(exist_ok=True)
def save01(name, arr01): cv2.imwrite(str(outdir / name), (np.clip(arr01,0,1)*255).astype(np.uint8))
save01("p3_bone_original.png", bone)
save01("p3_bilateral.png", bil_out)
save01("p3_nagao_matsuyama.png", nagao_out)
print(f"[INFO] Saved P3 images to: {outdir.resolve()}")
