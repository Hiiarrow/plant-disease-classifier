import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs("figures", exist_ok=True)

# 1. GENERATE CBAM ARCHITECTURE DIAGRAM
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.5)
ax.axis('off')

# Color Palette
c_input = '#1A365D'     # Navy
c_cam = '#0D9488'       # Teal
c_sam = '#2B6CB0'       # Blue
c_head = '#D69E2E'      # Gold
c_bg = '#F7FAFC'        # Soft gray

fig.patch.set_facecolor('white')

# Title
ax.text(5, 4.1, "Attention-MobileNetV3 (CBAM) Architectural Flow", fontsize=14, fontweight='bold', ha='center', color=c_input)

# Draw Boxes
# Input Feature Map
box_in = patches.FancyBboxPatch((0.5, 1.5), 1.6, 1.8, boxstyle="round,pad=0.1", ec=c_input, fc='#EBF8FF', lw=2)
ax.add_patch(box_in)
ax.text(1.3, 2.6, "Input Feature\nMap F", fontsize=10, fontweight='bold', ha='center', va='center', color=c_input)
ax.text(1.3, 1.8, "R^(576 x 7 x 7)", fontsize=8, ha='center', va='center', color='#4A5568')

# Channel Attention (CAM)
box_cam = patches.FancyBboxPatch((2.7, 1.5), 2.2, 1.8, boxstyle="round,pad=0.1", ec=c_cam, fc='#E6FFFA', lw=2)
ax.add_patch(box_cam)
ax.text(3.8, 2.7, "Channel Attention\nModule (CAM)", fontsize=10, fontweight='bold', ha='center', va='center', color=c_cam)
ax.text(3.8, 2.0, "AvgPool & MaxPool\n+ Shared MLP (r=16)\n=> M_c(F) in R^(576x1x1)", fontsize=8, ha='center', va='center', color='#2D3748')

# Spatial Attention (SAM)
box_sam = patches.FancyBboxPatch((5.5, 1.5), 2.2, 1.8, boxstyle="round,pad=0.1", ec=c_sam, fc='#EBF8FF', lw=2)
ax.add_patch(box_sam)
ax.text(6.6, 2.7, "Spatial Attention\nModule (SAM)", fontsize=10, fontweight='bold', ha='center', va='center', color=c_sam)
ax.text(6.6, 2.0, "Channel Avg & Max\n+ 7x7 Spatial Conv\n=> M_s(F') in R^(1x7x7)", fontsize=8, ha='center', va='center', color='#2D3748')

# Classifier Head
box_head = patches.FancyBboxPatch((8.3, 1.5), 1.4, 1.8, boxstyle="round,pad=0.1", ec=c_head, fc='#FEFCBF', lw=2)
ax.add_patch(box_head)
ax.text(9.0, 2.6, "Classifier\nHead", fontsize=10, fontweight='bold', ha='center', va='center', color='#744210')
ax.text(9.0, 1.8, "Linear(576->256)\nHardswish+Drop\nLinear(256->K)", fontsize=7.5, ha='center', va='center', color='#744210')

# Arrows
arrow_props = dict(boxstyle="rarrow,pad=0.1", fc='#A0AEC0', ec="none")
ax.annotate('', xy=(2.7, 2.4), xytext=(2.1, 2.4), arrowprops=dict(arrowstyle="-|>", lw=2, color=c_input))
ax.annotate('', xy=(5.5, 2.4), xytext=(4.9, 2.4), arrowprops=dict(arrowstyle="-|>", lw=2, color=c_cam))
ax.annotate('', xy=(8.3, 2.4), xytext=(7.7, 2.4), arrowprops=dict(arrowstyle="-|>", lw=2, color=c_sam))

plt.tight_layout()
cbam_path = os.path.join("figures", "cbam_architecture_diagram.png")
plt.savefig(cbam_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved: {cbam_path}")

# 2. GENERATE FIELD ADAPTATION PIPELINE DIAGRAM
fig, ax = plt.subplots(figsize=(10, 4.0), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.0)
ax.axis('off')

ax.text(5, 3.6, "3-Stage Real-World Field Adaptation Pipeline", fontsize=14, fontweight='bold', ha='center', color='#1A365D')

steps = [
    ("Stage 0: Unconstrained Field Image", "Raw Kashmiri Orchard Photo\nSoil, Glare, Shadow Clutter", '#E2E8F0', '#2D3748', 0.4),
    ("Stage 1: Foliar Leaf ROI Isolation", "HSV/Lab Masking\nSoil & Branch Removal", '#E6FFFA', '#0D9488', 2.8),
    ("Stage 2: 8x Patch Super-Sampling", "8 Overlapping Crops\nN = 2,680 Patches (+800%)", '#EBF8FF', '#2B6CB0', 5.2),
    ("Stage 3: 5-Crop Test-Time Aug", "5 Spatial Crops/Flips Logits\n76.19% Field Accuracy (+64.73%)", '#FEFCBF', '#744210', 7.6)
]

for title, desc, bg_c, text_c, x_pos in steps:
    box = patches.FancyBboxPatch((x_pos, 0.8), 2.0, 2.3, boxstyle="round,pad=0.1", ec=text_c, fc=bg_c, lw=2)
    ax.add_patch(box)
    ax.text(x_pos + 1.0, 2.6, title, fontsize=9.5, fontweight='bold', ha='center', va='center', color=text_c)
    ax.text(x_pos + 1.0, 1.6, desc, fontsize=8, ha='center', va='center', color='#4A5568')

for x in [2.4, 4.8, 7.2]:
    ax.annotate('', xy=(x + 0.4, 1.95), xytext=(x, 1.95), arrowprops=dict(arrowstyle="-|>", lw=2, color='#4A5568'))

plt.tight_layout()
pipeline_path = os.path.join("figures", "field_adaptation_pipeline.png")
plt.savefig(pipeline_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved: {pipeline_path}")
