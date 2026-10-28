# DeltaFuse: Dual-Stream Cloud Removal & Multispectral Super-Resolution for Tropical Deltas

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Earth Engine](https://img.shields.io/badge/Google_Earth_Engine-API-0F9D58.svg?style=flat&logo=googleearth)](https://earthengine.google.com/)
[![Copernicus](https://img.shields.io/badge/Copernicus-Sentinel--1%20%26%20Sentinel--2-003399.svg?style=flat)](https://sentinels.copernicus.eu/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Backbone](https://img.shields.io/badge/Backbone-ECRformer--Light%20(1.56M%20Params)-brightgreen.svg)]()
[![Target Domain](https://img.shields.io/badge/Target%20Domain-Ganges--Brahmaputra--Meghna%20Delta-blue.svg)]()
[![Status](https://img.shields.io/badge/Status-Active%20Research%20%26%20Development-orange.svg)]()

> **Author:** Md Abu Sayem Sarker  
> **Affiliation:** ClimateScholars Org • Department of Software Engineering, Daffodil International University  
> **Contact:** [LinkedIn](https://www.linkedin.com/in/mdabusayemsarker/)  
> **Research Track:** AI for Earth Observation (AI4EO) • Multi-Modal Remote Sensing

---

> [!NOTE]
> 🛰️ **Active Research & Development Notice:**  
> This repository is under active development as part of an upcoming research preprint. The codebase implements an end-to-end multi-modal Earth Observation (EO) pipeline designed for cloud penetration and spatial super-resolution in tropical monsoon deltas using European Space Agency (ESA) **Copernicus Sentinel-1 SAR** and **Sentinel-2 MSI** satellite data.

---

## 🌊 1. Scientific Motivation & Problem Formulation

In low-lying tropical delta systems such as the **Ganges-Brahmaputra-Meghna (GBM) delta**, persistent monsoon cloud cover obscures optical satellite instruments for more than **5 to 6 months every year**. Consequently, standard optical time-series monitoring completely collapses during critical flood disaster response, water-logging assessments, and crop-yield estimation cycles.

Synthetic Aperture Radar (SAR) sensors like **Sentinel-1** emit C-band microwaves ($\lambda \approx 5.6\text{ cm}$) that penetrate cloud layers, rain, and haze unimpeded. However, SAR backscatter conveys surface roughness and dielectric moisture properties rather than optical surface reflectance.

**DeltaFuse** bridges this physical divide via a two-stage multimodal framework:
1. **Stage 1 (Cloud Removal - Active Development):** Fuses contemporaneous cloudy Sentinel-2 (B2, B3, B4, B8 at 10m) and Sentinel-1 SAR (VV + VH in calibrated dB) to reconstruct clean 10m optical surface reflectance using **ECRformer-Light**—a lightweight transformer with Cross-Covariance Attention ($\mathcal{O}(NC^2)$ complexity) and Semantic-Decoupled Feature Learning (SDFL).
2. **Stage 2 (Multispectral Super-Resolution - In Pipeline):** Upscales reconstructed 10m multispectral bands to **5m analysis-ready imagery** using a spectrally regularized generator to preserve radiometric integrity without color drift.

```
                              [ SATELLITE ACQUISITION ]
                 Contemporaneous Copernicus Sentinel-1 & Sentinel-2
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
      Sentinel-2 MSI (Cloudy)                         Sentinel-1 GRD (SAR)
    Bands: B2, B3, B4, B8 (10m)                    Polarizations: VV, VH (10m)
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
           ═════════════════════════════════════════════════════════════
           STAGE 1: DUAL-STREAM CLOUD REMOVAL (DeltaFuse Backbone)
           ═════════════════════════════════════════════════════════════
           • Modality-Specific Parallel 7x7 Shallow Feature Embeddings
           • Cross-Covariance Attention (XCA, O(N·C²)) (1.56M Parameters)
           • Multi-Dilation Window Attention (MDWA, Dilations d=[1,2,3,4])
           • Semantic-Decoupled Feature Learning (SDFL Loss)
                                         │
                                         ▼
                          [ 10m Cloud-Free Optical Grid ]
                                         │
                                         ▼
           ═════════════════════════════════════════════════════════════
           STAGE 2: MULTISPECTRAL SUPER-RESOLUTION (OpenSR Pipeline)
           ═════════════════════════════════════════════════════════════
           • Spectrally-Aware 2× Spatial Upscaling (10m ➔ 5m)
           • Radiometric Preservation (StableSAM Angular Loss)
           • Narrow Riverbank & Agricultural Field Plot Delineation
                                         │
                                         ▼
                               [ FINAL 5m OUTPUT ]
                      Analysis-Ready Surface Reflectance
```

---

## ⚡ 2. Core Architectural Principles

### A. Modality-Specific Shallow Embedding (Section 3.2.1)
Optical reflectance (absorption-driven) and SAR backscatter (roughness-driven) belong to orthogonal physical manifolds. Early channel concatenation forces the network to reconcile contradictory statistics in initial layers. DeltaFuse processes optical and SAR inputs through independent $7\times 7$ convolutions before concatenating feature representations:
$$F_{\text{shallow}} = \text{Concat}\left(\text{Conv}_{\text{opt}}^{7\times 7}(X_{\text{opt}}), \, \text{Conv}_{\text{sar}}^{7\times 7}(X_{\text{sar}})\right)$$

### B. Cross-Covariance Attention (XCA, $\mathcal{O}(NC^2)$)
Standard self-attention computes an $N \times N$ spatial attention matrix ($N = H \times W$), scaling as $\mathcal{O}(N^2 C)$ which leads to extreme GPU memory exhaustion on high-resolution satellite scenes. DeltaFuse transposes attention across feature channels:
$$\text{XCAttention}(Q, K, V) = V \cdot \text{Softmax}\left( \frac{K^T Q}{\tau} \right)$$
where $\tau$ is a learnable per-head temperature parameter. This restricts complexity to $\mathcal{O}(NC^2)$, achieving **linear scaling in spatial resolution**.

### C. Multi-Dilation Window Attention (MDWA)
To capture multi-scale spatial context (road networks, river boundaries, and irregular cloud margins), MDWA samples keys and values at parallel dilation rates ($d \in [1, 2, 3, 4]$) across $3\times 3$ sparse windows, preconditioned with Conditional Positional Encoding (CPE via $3\times 3$ DWConv).

### D. Semantic-Decoupled Feature Learning (SDFL Loss)
Direct mapping from cloudy inputs to clear reflectance is ill-posed. DeltaFuse factorizes the objective into latent scene structure ($S$) and surface texture:
* **Encoder:** Guided by **Structural SSIM Loss** ($\mathcal{R}_{\text{enc}} = \mathbb{E}_k [1 - \text{SSIM}(\hat{Y}_{\text{enc}}^{(k)}, Y^{(k)})]$).
* **Decoder:** Guided by **Textural $L_1$ Loss** ($\mathcal{R}_{\text{dec}} = \mathbb{E}_k [\|\hat{Y}_{\text{dec}}^{(k)} - Y^{(k)}\|_1]$).
* **Total Objective:**
$$\mathcal{L} = 0.9 \cdot L_1(\hat{Y}, Y) + 0.1 \cdot (1 - \text{SSIM}(\hat{Y}, Y)) + 0.05 \cdot \mathcal{R}_{\text{enc}} + 0.05 \cdot \mathcal{R}_{\text{dec}}$$

### E. Spatial Data-Leakage Free Partitioning
* **Scene-Level & Geographic Splitting:** Satellite footprints are partitioned geographically into `train/` (65%), `val/` (20%), and `test/` (15%) **before tiling**. Overlapping patches from identical scenes never cross sets, preventing artificially inflated accuracy.
* **Geodetic Integrity:** Automatically assigns Projected UTM Zones based on longitude:
  * Western Delta ($\text{Lon} < 90^\circ\text{E}$): `EPSG:32645` (UTM Zone 45N).
  * Eastern Delta ($\text{Lon} \ge 90^\circ\text{E}$): `EPSG:32646` (UTM Zone 46N).

---

## 📂 3. Repository Structure

```
deltafuse-cloud-removal-super-resolution/
├── README.md                              # Scientific documentation and quick start guide
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
├── gee/
│   ├── gee_harvester_v2.py                # AOI-Adaptive Planetary Harvester (UTM 45N/46N)
│   └── gee.ipynb                          # Interactive Earth Engine query & temporal join
├── preprocessing/
│   ├── dataset.py                         # PyTorch Dataset supporting dual-target evaluation
│   ├── spatial_splitter_and_tiler.py      # Spatial data-leakage free scene splitting & tiling
│   └── preprocessing_walkthrough.ipynb    # Interactive step-by-step preprocessing notebook
├── models/
│   ├── ecrformer_light.py                 # Full paper-accurate ECRformer-Light & SDFL Loss
│   ├── ecrformer_dualstream.py            # Dual-stream encoder backbone variant
│   └── ecrformer_light_walkthrough.ipynb  # Interactive architecture & layer verification notebook
└── train_and_evaluate.py                  # End-to-end training, validation & benchmark harness
```

---

## 🗺️ 4. Research Roadmap & Milestones

- [x] **Milestone 1:** AOI-Adaptive Google Earth Engine Planetary Harvester for Copernicus Sentinel-1 & Sentinel-2.
- [x] **Milestone 2:** Spatial Data-Leakage Free Geographical Partitioning & 10m Affine Tiling Pipeline.
- [x] **Milestone 3:** Interactive Preprocessing Walkthrough Notebook with multi-modal visualizations (`preprocessing/preprocessing_walkthrough.ipynb`).
- [x] **Milestone 4:** Full Implementation of Paper-Accurate ECRformer-Light Backbone (1.56M parameters, XCA + MDWA + SGFN) (`models/ecrformer_light.py`).
- [x] **Milestone 5:** Interactive Architecture & Gradient Verification Walkthrough Notebook (`models/ecrformer_light_walkthrough.ipynb`).
- [ ] **Milestone 6:** Full Training & Validation on Multi-Seasonal Bangladesh Study Zones (Khulna Delta, Sylhet Haor Basin, Coastal Zone).
- [ ] **Milestone 7:** Comparative Benchmarking against Baselines (SEN12MS-CR benchmarks, Pix2Pix, GLF-CR, ECRformer base).
- [ ] **Milestone 8:** Stage 2 Integration: Multispectral 2× Super-Resolution Pipeline (OpenSR / SRGAN).
- [ ] **Milestone 9:** Manuscript preparation and preprint submission to **TechRxiv / EarthArXiv / arXiv**.

---

## 🚀 5. Quick Start & Developer Guide

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/deltafuse-cloud-removal-super-resolution.git
cd deltafuse-cloud-removal-super-resolution

# Create and activate Python virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Step 2: Google Earth Engine Authentication
```bash
python -c "import ee; ee.Authenticate(); ee.Initialize()"
```

### Step 3: Harvest Paired Sentinel-1 & Sentinel-2 Scenes
```bash
python gee/gee_harvester_v2.py
```
* Queries cloudy Sentinel-2 scenes, pairs the nearest temporal Sentinel-1 SAR acquisition ($\Delta t \le 3\text{ days}$), and exports 8-channel 10m GeoTIFFs (`B2, B3, B4, B8, cloud_prob, cloud_mask, VV, VH`) to Google Drive.

### Step 4: Preprocessing & Spatial Tiling (Zero Data Leakage)
```bash
python preprocessing/spatial_splitter_and_tiler.py
```
* Validates common 10m UTM affine grid, normalizes optical $[0, 1]$ and SAR $[-25, 0]\text{ dB} \rightarrow [-1, 1]$, and extracts $256 \times 256$ `.npz` patches into `data/patches_256/{train, val, test}/`.
* Or explore step-by-step in `preprocessing/preprocessing_walkthrough.ipynb`.

### Step 5: Verify Model Architecture
```bash
python models/ecrformer_light.py
```
* Verifies forward and backward gradient passes for the 1.56M parameter ECRformer-Light backbone and SDFL multi-scale loss.
* Or explore step-by-step in `models/ecrformer_light_walkthrough.ipynb`.

### Step 6: Train & Benchmark Model
```bash
python train_and_evaluate.py
```

---

## 📊 6. Evaluation Protocol & Target Metrics

Evaluation adheres to a rigorous dual-tier protocol designed for peer-reviewed remote sensing standards:

1. **Synthetic Cloud Occlusion Benchmark (Exact Ground Truth):** Clear-sky Sentinel-2 acquisitions occluded with multi-scale fractal cloud noise to measure exact mathematical reconstruction fidelity.
2. **Real-World Empirical Evaluation (Proxy Reference):** Real monsoon cloudy scenes paired with contemporaneous SAR and validated against near-date seasonal clear acquisitions.

| Metric | Target Standard | Primary Focus |
|:---|:---:|:---|
| **PSNR (dB)** | $\ge 30.0\text{ dB}$ | Pixel-level reconstruction fidelity |
| **SSIM** | $\ge 0.88$ | Structural and edge perception preservation |
| **SAM (Spectral Angle)** | $\le 4.5^\circ$ | Vector angle preservation across multispectral bands |
| **NRMSE** | $\le 0.08$ | Normalized root mean squared error |
| **Backbone Parameters** | $\le 2.0\text{ M}$ | **1.56M parameters** (Deployable on standard workstations) |

---

## 📜 Academic References & Acknowledgments

This research builds upon foundational principles from:
* **ECRformer:** Zhang, Z., et al. (2026). *ECRformer: An efficient cloud removal Transformer with semantic-decoupled learning for multimodal satellite imagery.* ISPRS Journal of Photogrammetry and Remote Sensing, 237, 323–338.
* **SEN12MS-CR:** Schmitt, M., et al. (2020). *SEN12MS-CR: A dataset for cloud removal from multi-modal Sentinel-1 and Sentinel-2 satellite data.*
* **OpenSR:** Aybar, C., et al. (2023). *OpenSR-test: An open-source benchmark for super-resolution in satellite imagery.* In IEEE/CVF CVPRW.

If you find this research useful in your work, please cite:

```bibtex
@misc{sarker2026deltafuse,
  author       = {Md Abu Sayem Sarker},
  title        = {DeltaFuse: Dual-Stream Cloud Removal and Multispectral Super-Resolution for Tropical Deltas},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/your-username/deltafuse-cloud-removal-super-resolution}},
  note         = {Active research prototype for multimodal Earth Observation}
}
```

---

## 📄 License & Attribution
* Released under the **MIT License** — see [LICENSE](LICENSE) for details.
* Satellite imagery courtesy of the **European Union Copernicus Programme** and the **European Space Agency (ESA)**.
* Planetary computing infrastructure powered by **Google Earth Engine**.
