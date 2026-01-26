import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
import satlaspretrain_models
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
from matplotlib import cm

"""
Satlas Feature Extraction + Clustering for Building Segmentation
Using the sliding window approach with Satlas pretrained models.
"""

def extract_patches_with_overlap(img, window_size=512, stride=None, padding_mode='reflect'):
    """Extract overlapping patches from a large image."""
    if stride is None:
        stride = window_size // 2
    
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    
    pad_h = (window_size - h % stride) % stride if h > window_size else window_size - h
    pad_w = (window_size - w % stride) % stride if w > window_size else window_size - w
    
    if len(img_array.shape) == 3:
        img_padded = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode=padding_mode)
    else:
        img_padded = np.pad(img_array, ((0, pad_h), (0, pad_w)), mode=padding_mode)
    
    h_padded, w_padded = img_padded.shape[:2]
    
    patches = []
    positions = []
    
    for i in range(0, h_padded - window_size + 1, stride):
        for j in range(0, w_padded - window_size + 1, stride):
            patch = img_padded[i:i+window_size, j:j+window_size]
            patches.append(Image.fromarray(patch))
            positions.append((i, j))
    
    grid_info = {
        'original_size': (h, w),
        'padded_size': (h_padded, w_padded),
        'window_size': window_size,
        'stride': stride,
        'n_rows': len(range(0, h_padded - window_size + 1, stride)),
        'n_cols': len(range(0, w_padded - window_size + 1, stride))
    }
    
    return patches, positions, grid_info


def extract_satlas_features_sliding_window(img, model, device, window_size=512, 
                                          stride=None, batch_size=4, feature_level=1):
    """
    Extract Satlas features using sliding window approach.
    
    Args:
        feature_level: Which FPN level to use (0=finest/largest, 4=coarsest/smallest)
                      0: 1x downsampling (512x512)
                      1: 4x downsampling (128x128) - recommended for balance
                      2: 8x downsampling (64x64)
                      3: 16x downsampling (32x32)
                      4: 32x downsampling (16x16)
    """
    print(f"Extracting patches from image of size {img.size}...")
    patches, positions, grid_info = extract_patches_with_overlap(
        img, window_size=window_size, stride=stride
    )
    
    print(f"Created {len(patches)} patches ({grid_info['n_rows']}x{grid_info['n_cols']} grid)")
    
    # Preprocessing for Satlas
    transform = transforms.Compose([
        transforms.ToTensor(),  # Already in [0, 1] range
    ])
    
    all_features = []
    
    for i in tqdm(range(0, len(patches), batch_size), desc="Processing patches"):
        batch_patches = patches[i:i+batch_size]
        
        # Preprocess batch
        batch_tensors = torch.stack([transform(patch) for patch in batch_patches]).to(device)
        
        # Extract features
        with torch.no_grad():
            features = model(batch_tensors)
            
            # Satlas with FPN returns multi-scale features
            # Handle different return types: dict, list, or tensor
            if isinstance(features, dict):
                # Dict of feature maps at different scales
                # Try common keys, fallback to first value
                for key in ['res2', 'layer1', 'stage1']:
                    if key in features:
                        feat = features[key]
                        break
                else:
                    feat = list(features.values())[0]
            elif isinstance(features, (list, tuple)):
                # List of feature maps at different scales
                # Use the specified feature level
                feat = features[feature_level]
            else:
                # Single tensor
                feat = features
            
            # Ensure it's a tensor
            if not isinstance(feat, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(feat)}")
        
        # Features shape: (batch, channels, h_feat, w_feat)
        # Reshape to (batch, h_feat, w_feat, channels) for clustering
        batch_features = feat.permute(0, 2, 3, 1).cpu().numpy()
        all_features.append(batch_features)
    
    # Concatenate all batches
    all_features = np.concatenate(all_features, axis=0)
    
    # Determine actual downsample factor from feature size
    batch_size, h_feat, w_feat, channels = all_features.shape
    actual_downsample = window_size // h_feat
    
    print(f"Feature dimensions per patch: {h_feat}x{w_feat}, downsample factor: {actual_downsample}")
    
    # Stitch patches together
    feature_map = stitch_satlas_features(all_features, positions, grid_info, 
                                        downsample_factor=actual_downsample)
    
    return feature_map, grid_info


def stitch_satlas_features(patch_features, positions, grid_info, downsample_factor):
    """Stitch Satlas patch features into a single feature map."""
    h_padded, w_padded = grid_info['padded_size']
    window_size = grid_info['window_size']
    D = patch_features.shape[-1]
    
    # Output feature map dimensions (downsampled by model)
    h_feat = h_padded // downsample_factor
    w_feat = w_padded // downsample_factor
    
    feature_map = np.zeros((h_feat, w_feat, D), dtype=np.float32)
    weight_map = np.zeros((h_feat, w_feat), dtype=np.float32)
    
    patches_per_window = window_size // downsample_factor
    
    for patch_feat, (row, col) in zip(patch_features, positions):
        row_feat = row // downsample_factor
        col_feat = col // downsample_factor
        
        h_slice = slice(row_feat, row_feat + patches_per_window)
        w_slice = slice(col_feat, col_feat + patches_per_window)
        
        # Simple averaging (can add weighting if needed)
        feature_map[h_slice, w_slice] += patch_feat
        weight_map[h_slice, w_slice] += 1.0
    
    # Normalize by weights
    feature_map = feature_map / (weight_map[..., np.newaxis] + 1e-8)
    
    # Crop to original size
    h_orig, w_orig = grid_info['original_size']
    h_orig_feat = h_orig // downsample_factor
    w_orig_feat = w_orig // downsample_factor
    feature_map = feature_map[:h_orig_feat, :w_orig_feat]
    
    return feature_map



def segment_with_satlas(img_path, device='cuda', model_checkpoint="Aerial_SwinB_SI",
                       window_size=512, stride=256, n_clusters=5, batch_size=4,
                       feature_level=1):
    """
    Complete pipeline for building segmentation using Satlas.
    
    Args:
        feature_level: Which feature pyramid level to use
                      0: Finest (512x512) - most detail, slower
                      1: Medium (128x128) - good balance (recommended)
                      2: Coarse (64x64) - faster, less detail
    """
    # Load Satlas model
    print("Loading Satlas pretrained model...")
    weights_manager = satlaspretrain_models.Weights()
    model = weights_manager.get_pretrained_model(model_checkpoint, fpn=True)
    model = model.to(device)
    model.eval()
    
    # Load image
    img = Image.open(img_path).convert("RGB")
    print(f"Processing image of size: {img.size}")
    
    # Extract features with sliding window
    feature_map, grid_info = extract_satlas_features_sliding_window(
        img, model, device,
        window_size=window_size, stride=stride, batch_size=batch_size,
        feature_level=feature_level
    )
    
    print(f"Feature map shape: {feature_map.shape}")
    
    # Flatten features for clustering
    h_feat, w_feat, D = feature_map.shape
    features_flat = feature_map.reshape(-1, D)
    
    # K-means clustering
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(features_flat)
    
    # Reshape to spatial map
    label_map = labels.reshape(h_feat, w_feat)
    
    # Upsample to original image size
    h_orig, w_orig = grid_info['original_size']
    label_map_upsampled = cv2.resize(
        label_map.astype(np.float32),
        (w_orig, h_orig),
        interpolation=cv2.INTER_NEAREST
    )
    
    return label_map_upsampled, label_map, img


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_path = "rgb_maps/davis_rgb.tif"
    
    print("=" * 60)
    print("Building Segmentation with Satlas Pretrained Model")
    print("=" * 60)
  
    label_map_full, label_map_patches, img = segment_with_satlas(
        img_path=img_path,
        device=device,
        model_checkpoint="Aerial_SwinB_SI",
        window_size=512,
        stride=256,
        n_clusters=2,
        batch_size=4,
        feature_level=1
    )

    # Normalize labels to [0,1] for colormap lookup
    norm = (label_map_full - label_map_full.min()) / (label_map_full.max() - label_map_full.min() + 1e-8)
    cmap = cm.get_cmap('viridis')
    colored = cmap(norm)[:, :, :3]  # RGB floats in [0,1]
    colored_uint8 = (colored * 255).astype(np.uint8)

    # Convert RGB -> BGR for OpenCV
    colored_bgr = colored_uint8[..., ::-1]

    os.makedirs('results', exist_ok=True)
    output_path = 'results/davis_rgb_segmented.png'
    cv2.imwrite(output_path, colored_bgr)
    print(f"Saved segmentation to: {output_path}")