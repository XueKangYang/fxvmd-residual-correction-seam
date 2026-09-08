# f-x VMD + Residual-Correction Learning for Seismic Denoising

This repository provides a MATLAB implementation of a seismic random-noise attenuation workflow that combines f-x domain variational mode decomposition (f-x VMD) with a two-channel residual-correction network.

The public version is organized for training from SEAM synthetic seismic data only. Private field datasets and previously trained model parameters are not included.

## Main idea

For each noisy seismic section `D`, the workflow first applies f-x VMD to obtain a preliminary denoised section `F`. The f-x VMD residual is computed as

```text
R_fx = D - F
```

Then `[R_fx, F]` is concatenated as a two-channel input to a U-Net/DnCNN-style residual network. The network predicts the residual noise `R_pred`, and the final denoised result is

```text
D_denoised = D - R_pred
```

## Repository structure

```text
fxvmd_residual_correction_seam_github/
├── README.md
├── LICENSE
├── NOTICE.md
├── .gitignore
├── scripts/
│   ├── main_train_seam.m
│   └── generate_seam_dataset.m
├── functions/
│   ├── fxvmd.m
│   ├── build_unet_dncnn.m
│   ├── readSEGY_patch_noToolbox.m
│   ├── getSEGYInfo_noToolbox.m
│   ├── add_awgn_measured.m
│   ├── normalize_seismic_patch.m
│   ├── make_two_channel_pairs.m
│   ├── compute_snr_db.m
│   └── plot_seismic_section.m
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
│       └── README.md
├── models/
│   └── README.md
├── outputs/
│   ├── figures/
│   │   └── README.md
│   └── metrics/
│       └── README.md
└── examples/
    └── run_quick_demo_without_segy.m
```

## Requirements

Tested workflow assumptions:

- MATLAB with Deep Learning Toolbox.
- MATLAB `vmd` function support, usually provided by Signal Processing Toolbox.
- No MATLAB SEGY toolbox is required; this repository includes a simple SEGY reader for common SEGY format codes.
- GPU is optional. If no GPU is available, set `executionEnvironment = 'cpu'` in `scripts/main_train_seam.m`.

## Data preparation

Place the SEAM time-domain SEGY file under:

```text
data/raw/SEAM_Interpretation_Challenge_1_Time.sgy
```

or edit this line in `scripts/main_train_seam.m`:

```matlab
cfg.segyPath = fullfile(projectRoot, 'data', 'raw', 'SEAM_Interpretation_Challenge_1_Time.sgy');
```

## How to run

From MATLAB, open the repository root and run:

```matlab
run('scripts/main_train_seam.m')
```

The script will:

1. read random 2-D sections from the SEAM SEGY file;
2. normalize clean sections;
3. add measured AWGN at random SNR levels;
4. apply the separately provided `functions/fxvmd.m` to each noisy section;
5. build two-channel inputs `[R_fx, F]`;
6. train the residual-correction network from scratch;
7. evaluate on held-out SEAM sections;
8. save model, metrics, and example figures.

## Important privacy note

This public version does not include private field seismic datasets or previously trained parameters. Generated `.mat`, `.sgy`, model, cache, and output files are ignored by `.gitignore` by default.

## License note

The `functions/fxvmd.m` file follows the GPL notice provided in the original source header. Because this repository distributes that file, the repository is released under GPL-compatible terms. See `LICENSE` and `NOTICE.md`.
