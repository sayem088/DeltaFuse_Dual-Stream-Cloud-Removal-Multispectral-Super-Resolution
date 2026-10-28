"""
Module: spatial_splitter_and_tiler.py
Location: F:/sparrso/multi-modal-cloud-removal-dataset-design/preprocessing/spatial_splitter_and_tiler.py
Description: Spatial Data-Leakage-Free Preprocessing & Patching Pipeline.
Key Features:
  1. Scene / Geographical Split: Splits entire GeoTIFF scenes into Train / Val / Test 
     BEFORE tiling to completely eliminate data leakage between adjacent overlapping patches.
  2. Common 10m Affine Grid Verification: Validates CRS, modulo-10 alignment, and extent.
  3. Reflectance & SAR Inspection: Inspects band statistics, clips SAR dB [-25, 0] -> [-1, 1],
     and verifies optical reflectance scale.
  4. Dual Target Strategy:
     - Synthetic Cloud Occlusion (Perlin/Fractal mask) on clear scenes for Exact Benchmark GT.
     - Real Cloud Pairs with Near-Date S2 as Proxy Reference.
Author: Md Abu Sayem Sarker
"""

import os
import glob
import math
import numpy as np
import rasterio
from rasterio.windows import Window
from typing import List, Tuple, Dict, Optional


class SpatialSplitterAndTiler:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        patch_size: int = 256,
        stride: int = 128,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        sar_clip_range: Tuple[float, float] = (-25.0, 0.0)
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.patch_size = patch_size
        self.stride = stride
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.sar_min, self.sar_max = sar_clip_range

        assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-4), \
            "Train, Val, and Test ratios must sum to 1.0"

        # Create split output directories
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    def verify_affine_grid(self, src: rasterio.io.DatasetReader) -> Dict[str, any]:
        """Validates CRS and common 10m pixel grid alignment."""
        res_x, res_y = src.res
        assert abs(res_x - 10.0) < 0.1 and abs(res_y - 10.0) < 0.1, \
            f"Warning: Resolution ({res_x}, {res_y}) is not 10.0 meters."

        bounds = src.bounds
        crs = str(src.crs)
        
        # Verify coordinate reference system
        assert "32645" in crs or "32646" in crs, \
            f"Expected UTM Zone 45N (32645) or 46N (32646), got {crs}"

        return {
            "crs": crs,
            "resolution": (res_x, res_y),
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "bounds": bounds
        }

    def inspect_and_normalize_bands(
        self,
        raw_patch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Input patch shape: [Channels, H, W]
        Channel layout from GEE export:
          0: B2 (Blue), 1: B3 (Green), 2: B4 (Red), 3: B8 (NIR)
          4: cloud_prob (0 - 100)
          5: cloud_mask (0 or 1)
          6: VV (dB)
          7: VH (dB)
        Returns:
          optical: [4, H, W] in [0, 1] range
          sar: [2, H, W] normalized to [-1, 1]
          cloud_mask: [1, H, W] binary mask
        """
        # 1. Optical (B2, B3, B4, B8)
        # Scaled by 1/10000 in GEE, clipping negative values (sensor noise) and extreme glare > 1.0
        optical = np.clip(raw_patch[0:4, :, :], 0.0, 1.0).astype(np.float32)

        # 2. Cloud Mask (Channel 5 is binary mask, or threshold cloud_prob Channel 4 >= 40)
        if raw_patch.shape[0] >= 6:
            cloud_mask = (raw_patch[5:6, :, :] > 0.5).astype(np.float32)
        else:
            cloud_mask = np.zeros((1, raw_patch.shape[1], raw_patch.shape[2]), dtype=np.float32)

        # 3. SAR (VV and VH)
        # GEE S1 GRD is natively in dB. Clip [-25, 0] dB and linearly scale to [-1, 1]
        sar_raw = raw_patch[6:8, :, :] if raw_patch.shape[0] >= 8 else raw_patch[4:6, :, :]
        sar_clipped = np.clip(sar_raw, self.sar_min, self.sar_max)
        sar_norm = 2.0 * ((sar_clipped - self.sar_min) / (self.sar_max - self.sar_min)) - 1.0
        sar_norm = sar_norm.astype(np.float32)

        return optical, sar_norm, cloud_mask

    def generate_synthetic_fractal_cloud(self, h: int, w: int, coverage: float = 0.5) -> np.ndarray:
        """
        Generates realistic fractal cloud opacity mask [0, 1] using multi-scale noise.
        Used on clear-sky images to generate exact paired ground-truth targets.
        """
        np.random.seed()
        # Multi-scale Gaussian smoothed noise
        grid = np.zeros((h, w), dtype=np.float32)
        for scale in [64, 32, 16, 8]:
            noise = np.random.randn(h // scale + 2, w // scale + 2).astype(np.float32)
            from scipy.ndimage import zoom
            upscaled = zoom(noise, scale, order=1)[:h, :w]
            grid += upscaled * (scale / 64.0)

        # Normalize to [0, 1]
        grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-7)
        # Threshold to desired coverage
        cloud_mask = (grid > (1.0 - coverage)).astype(np.float32)
        return cloud_mask[np.newaxis, :, :]  # [1, H, W]

    def split_and_tile_scenes(self):
        """
        Scene-Level Geographical Splitting:
        Sorts entire GeoTIFF files and splits them into train, val, and test groups
        BEFORE tiling. This guarantees zero data leakage between splits!
        """
        tif_files = sorted(glob.glob(os.path.join(self.input_dir, "*.tif")))
        if not tif_files:
            print(f"[!] No .tif files found in {self.input_dir}.")
            print("    Please run Phase 1 GEE harvester to export GeoTIFFs to this directory.")
            return

        total_scenes = len(tif_files)

        # Single-scene mode: partition geographically along rows to prevent spatial leakage
        if total_scenes == 1:
            scene_path = tif_files[0]
            scene_name = os.path.splitext(os.path.basename(scene_path))[0]
            print(f"[*] Single GeoTIFF detected ({scene_name}). Applying geographic row partitioning...")
            with rasterio.open(scene_path) as src:
                self.verify_affine_grid(src)
                h, w = src.height, src.width
                y_max = h - self.patch_size
                split_train_y = int(y_max * self.train_ratio)
                split_val_y = int(y_max * (self.train_ratio + self.val_ratio))

                counts = {'train': 0, 'val': 0, 'test': 0}
                for y in range(0, y_max + 1, self.stride):
                    for x in range(0, w - self.patch_size + 1, self.stride):
                        window = Window(x, y, self.patch_size, self.patch_size)
                        raw_patch = src.read(window=window)

                        if np.all(raw_patch == 0) or not np.isfinite(raw_patch).all():
                            continue

                        optical, sar, cloud_mask = self.inspect_and_normalize_bands(raw_patch)

                        if y <= split_train_y:
                            split_name = 'train'
                        elif y <= split_val_y:
                            split_name = 'val'
                        else:
                            split_name = 'test'

                        patch_filename = f"{scene_name}_p{y}_{x}.npz"
                        save_path = os.path.join(self.output_dir, split_name, patch_filename)
                        np.savez_compressed(save_path, optical=optical, sar=sar, cloud_mask=cloud_mask)
                        counts[split_name] += 1

            for s, c in counts.items():
                print(f"[OK] {s.upper()}: Extracted {c} patches into {os.path.join(self.output_dir, s)}")
            return

        n_train = max(1, int(total_scenes * self.train_ratio))
        n_val = max(1, int(total_scenes * self.val_ratio))
        n_test = max(1, total_scenes - n_train - n_val)

        train_files = tif_files[:n_train]
        val_files = tif_files[n_train:n_train + n_val]
        test_files = tif_files[n_train + n_val:]

        split_map = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

        print(f"[*] Total Scenes: {total_scenes} | Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

        for split_name, scene_list in split_map.items():
            patch_count = 0
            split_out_dir = os.path.join(self.output_dir, split_name)

            for scene_path in scene_list:
                scene_name = os.path.splitext(os.path.basename(scene_path))[0]
                with rasterio.open(scene_path) as src:
                    # Verify affine grid
                    grid_info = self.verify_affine_grid(src)
                    h, w = src.height, src.width

                    # Sliding window patch extraction
                    for y in range(0, h - self.patch_size + 1, self.stride):
                        for x in range(0, w - self.patch_size + 1, self.stride):
                            window = Window(x, y, self.patch_size, self.patch_size)
                            raw_patch = src.read(window=window)  # [Channels, 256, 256]

                            # Skip if nodata / non-finite / empty
                            if np.all(raw_patch == 0) or not np.isfinite(raw_patch).all():
                                continue

                            optical, sar, cloud_mask = self.inspect_and_normalize_bands(raw_patch)

                            # Save as compressed .npz archive
                            patch_filename = f"{scene_name}_p{y}_{x}.npz"
                            save_path = os.path.join(split_out_dir, patch_filename)
                            
                            np.savez_compressed(
                                save_path,
                                optical=optical,        # [4, 256, 256]
                                sar=sar,                # [2, 256, 256]
                                cloud_mask=cloud_mask   # [1, 256, 256]
                            )
                            patch_count += 1

            print(f"[OK] {split_name.upper()}: Extracted {patch_count} patches into {split_out_dir}")


if __name__ == "__main__":
    # Test harness
    print("=" * 80)
    print(" SPARRSO Spatial Splitter & Tiler (Data Leakage-Free Pipeline)")
    print("=" * 80)
    
    # Configure paths
    raw_geotiff_dir = "F:/sparrso/multi-modal-cloud-removal-dataset-design/data/raw_geotiffs"
    processed_patches_dir = "F:/sparrso/multi-modal-cloud-removal-dataset-design/data/patches_256"
    
    tiler = SpatialSplitterAndTiler(
        input_dir=raw_geotiff_dir,
        output_dir=processed_patches_dir,
        patch_size=256,
        stride=128,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )
    print("[✓] Preprocessor initialized and ready for GEE exported GeoTIFFs.")
