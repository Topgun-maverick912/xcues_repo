import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import math
import gc
import matplotlib.pyplot as plt
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==========================================
# 0. SETTINGS & PATHS (ANONYMIZED)
# ==========================================
HF_TOKEN = "YOUR_HF_TOKEN_HERE"
IMAGE_PATH = "./sample_image.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running Fast 16-State XCues Analyzer on: {DEVICE}")

# ==========================================
# 1. CORE XCUES FUNCTIONS
# ==========================================
GEOM_PERM = [13, 8, 11, 6, 5, 0, 3, 14, 2, 15, 12, 9, 10, 7, 4, 1]

def preprocess_image(img_raw, size=224):
    img = np.array(img_raw.convert('RGB'))
    h, w = img.shape[:2]
    if h < w:
        new_h, new_w = size, int(w * size / h)
    else:
        new_h, new_w = int(h * size / w), size
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    start_y, start_x = (new_h - size) // 2, (new_w - size) // 2
    return img[start_y:start_y+size, start_x:start_x+size]

def get_feathered_mask(img_224, seg_model, device):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_448 = cv2.resize(img_224, (448, 448), interpolation=cv2.INTER_LINEAR)
    tensor_img = preprocess(img_448).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = seg_model(tensor_img)['out'][0]
    
    mask_binary_448 = output.argmax(0).byte().cpu().numpy() > 0
    mask_binary = cv2.resize(mask_binary_448.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST) > 0
    
    mask_ratio = mask_binary.mean()
    if mask_ratio < 0.05 or mask_ratio > 0.95:
        # Fallback to 70% center box if segmentation mask coverage is <5% or >95%
        mask_binary = np.zeros((224, 224), dtype=bool)
        ch, cw = int(224 * 0.7), int(224 * 0.7)
        mask_binary[(224-ch)//2:(224+ch)//2, (224-cw)//2:(224+cw)//2] = True
        
    mask_float = mask_binary.astype(np.float32)
    mask_feathered = cv2.GaussianBlur(mask_float, (21, 21), 0)
    return np.expand_dims(mask_feathered, axis=-1)

def suppress_geometry(img, grid_size=4):
    h, w, _ = img.shape
    ph, pw = h // grid_size, w // grid_size
    patches = [img[i*ph:(i+1)*ph, j*pw:(j+1)*pw] for i in range(grid_size) for j in range(grid_size)]
    patches = [patches[k] for k in GEOM_PERM]
    rows = [np.hstack(patches[r*grid_size:(r+1)*grid_size]) for r in range(grid_size)]
    return np.vstack(rows)

def apply_filters(img, active_cues, mask_3d):
    out = img.astype(np.float32)
    
    # Context Suppression (sigma=15)
    if active_cues[3] == 0: 
        bg = cv2.GaussianBlur(out, (99, 99), 15)
        out = (out * mask_3d) + (bg * (1.0 - mask_3d))
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)

    # Color (Spectral Appearance) Suppression
    if active_cues[2] == 0: 
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
    # Texture Suppression
    if active_cues[1] == 0: 
        out = cv2.bilateralFilter(out, d=11, sigmaColor=170, sigmaSpace=75)
        
    # Geometry Suppression (Deterministic Annealed Permutation)
    if active_cues[0] == 0: 
        out = suppress_geometry(out, grid_size=4)
        
    return out

ALL_STATES = [tuple(map(int, list(f"{i:04b}"))) for i in range(16)]

def calculate_shapley_from_dict(state_logits):
    shapley = [0.0, 0.0, 0.0, 0.0]
    for i in range(4):
        for state in ALL_STATES:
            if state[i] == 1:
                state_absent = list(state)
                state_absent[i] = 0
                marginal = state_logits[state] - state_logits[tuple(state_absent)]
                S = sum(state_absent)
                weight = math.factorial(S) * math.factorial(4 - S - 1) / math.factorial(4)
                shapley[i] += weight * marginal
    return shapley

# ==========================================
# 2. GENERATE AND VISUALIZE 16 PERMUTATIONS
# ==========================================
print("Loading and preprocessing image...")
try:
    raw_pil = Image.open(IMAGE_PATH).convert('RGB')
except FileNotFoundError:
    print(f"Error: Could not find image at {IMAGE_PATH}. Please provide a valid image.")
    exit()

img_224 = preprocess_image(raw_pil)
img_float = img_224.astype(np.float32) / 255.0

print("Generating semantic mask...")
seg_model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT).to(DEVICE)
seg_model.eval()
mask_3d = get_feathered_mask(img_224, seg_model, DEVICE)
del seg_model
torch.cuda.empty_cache()
gc.collect()

print("Generating exactly 16 permutations...")
batch_images = []
state_to_indices = {}

for i, state in enumerate(ALL_STATES):
    filtered = apply_filters(img_224, state, mask_3d)
    batch_images.append(filtered)
    state_to_indices[state] = [i]

print("Plotting the 16 states grid...")
fig_states, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.flatten()
cue_names = ['Geometry', 'Texture', 'Color', 'Context']

for idx, state in enumerate(ALL_STATES):
    ax = axes[idx]
    ax.imshow(batch_images[idx])
    ax.axis('off')
    
    active_cues = [cue_names[i] for i in range(4) if state[i] == 1]
    title_cues = "+".join(active_cues) if active_cues else "None"
    if state == (1, 1, 1, 1): title_cues = "All"
    
    ax.set_title(f"State: {state}\nActive: {title_cues}", fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('xcues_16_states_grid.png', bbox_inches='tight', dpi=200)
plt.close()

# ==========================================
# 3. RUN INFERENCE & SHAPLEY MATH
# ==========================================
batch_u8 = torch.from_numpy(np.stack(batch_images))

print("Running ResNet-50 inference...")
resnet_weights = models.ResNet50_Weights.IMAGENET1K_V1
model = models.resnet50(weights=resnet_weights).to(DEVICE)
model.eval()

imagenet_categories = resnet_weights.meta["categories"]

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

with torch.no_grad():
    x = batch_u8.to(DEVICE).permute(0, 3, 1, 2).float() / 255.0
    batch_logits = model((x - MEAN) / STD)
    
    baseline_indices = state_to_indices[(1, 1, 1, 1)]
    baseline_logits = batch_logits[baseline_indices[0]]
    pred_class = torch.argmax(baseline_logits).item()
    pred_class_name = imagenet_categories[pred_class].title()
    
    state_logits = {}
    for state in ALL_STATES:
        indices = state_to_indices[state]
        avg_logit = batch_logits[indices, pred_class].mean().item()
        state_logits[state] = avg_logit

shapley_logits = calculate_shapley_from_dict(state_logits)
pos_shapley = np.clip(shapley_logits, 0, None)
total_impact = max(pos_shapley.sum(), 1e-12)
pcts = 100 * pos_shapley / total_impact

print("\n" + "="*50)
print("XCues (Shapley) Attribution Results:")
print("="*50)
print(f"Geometry : {pcts[0]:.1f}%")
print(f"Texture  : {pcts[1]:.1f}%")
print(f"Color    : {pcts[2]:.1f}%")
print(f"Context  : {pcts[3]:.1f}%")
print("="*50 + "\n")

# ==========================================
# 4. COMPUTE GRAD-CAM & PLOT FINAL RESULT
# ==========================================
print("Computing Grad-CAM (layer4)...")
target_layers = [model.layer4[-1]] 
cam = GradCAM(model=model, target_layers=target_layers)

input_tensor = ((torch.from_numpy(img_float).to(DEVICE).permute(2, 0, 1).unsqueeze(0) - MEAN) / STD)
input_tensor.requires_grad_(True)

targets = [ClassifierOutputTarget(pred_class)]
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

print("Generating Final Plot...")
plt.style.use('seaborn-v0_8-whitegrid')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1, 1.2]})

ax1.imshow(img_224)
ax1.axis('off')
ax1.set_title(f"Input Image\n(Class: {pred_class_name})", fontsize=14, fontweight='bold', pad=10)

ax2.imshow(visualization)
ax2.axis('off')
ax2.set_title("Grad-CAM Attribution", fontsize=14, fontweight='bold', pad=10)

cues = ['Geometry', 'Texture', 'Color', 'Context']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax3.bar(cues, pcts, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_ylim(0, max(pcts) + 20)
ax3.set_ylabel('Cue Contribution (%)', fontsize=12, fontweight='bold')
ax3.set_title('XCues (Shapley) Attribution', fontsize=14, fontweight='bold', pad=10)

for bar in bars:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f"{bar.get_height():.1f}%", 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('xcues_comparison.png', format='png', bbox_inches='tight', dpi=300)

print("Success! Saved visualization outputs.")
