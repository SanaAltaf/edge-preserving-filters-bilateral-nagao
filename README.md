# Edge-Preserving Filters: Bilateral and Nagao–Matsuyama

This project implements and compares two **edge-preserving smoothing filters** on a grayscale medical-style image (e.g., bone scan):

- **Bilateral filter**
- **Nagao–Matsuyama filter**

Both filters aim to **reduce noise** while **preserving important edges and fine structures**, which is crucial in medical imaging, robotics, and other Edge AI applications.

---

## 📌 Project Overview

Classical smoothing (like Gaussian blur) removes noise but also blurs edges, which is often unacceptable in tasks where boundaries and structures matter.

This project focuses on two edge-preserving approaches:

### 🔹 Bilateral Filter
- Combines:
  - **Spatial distance** (nearby pixels)
  - **Intensity similarity** (similar pixel values)
- Smooths within homogeneous regions
- Keeps strong edges sharp

### 🔹 Nagao–Matsuyama Filter
- Uses a **5×5 window** divided into several overlapping **3×3 regions**
- Picks the region with the **smallest variance** (most homogeneous)
- Replaces the center pixel with that region’s mean
- Strongly preserves:
  - Thin structures
  - Edges
  - Boundaries in medical images

---

## 🧪 What the Script Does

The main script (one-file project):

- Loads a grayscale input image (e.g., bone scan)
- Applies:
  - Standard **bilateral filter**
  - **Nagao–Matsuyama filter**
- Displays or saves:
  - Original image
  - Bilateral-filtered image
  - Nagao-filtered image
- Optionally prints simple statistics (e.g., edge strength / variance)

---

## 📂 Folder Structure

```text
edge-preserving-filters-bilateral-nagao/
│
├── edge_preserving_filters.py      # main script: bilateral + Nagao
├── images/
│     ├── bone_scan_input.png       # input image
│     ├── bilateral_output.png      # bilateral-filtered output
│     └── nagao_output.png          # Nagao–Matsuyama output
├── README.md
└── requirements.txt

