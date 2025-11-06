# px-umbed

**Pixel-wise Micro Embeddings with Multi-Scale Patch Pyramids**

px-umbed is a neural network architecture that learns micro embeddings of each pixel's local neighborhood through a hierarchical image pyramid representation. By encoding the local patch context at multiple scales, the model creates rich, spatially-aware latent representations that capture both fine-grained details and broader contextual information.

## Overview

The model processes images by:
1. **Multi-scale patch extraction**: For each pixel, extracts concentric patches at multiple pyramid levels (e.g., 8×8, 16×16, 32×32)
2. **Hierarchical encoding**: Uses a U-Net encoder-decoder with VAE bottleneck to compress each pixel's local context into a compact latent vector
3. **Patch reconstruction**: Decodes each latent vector back into the original multi-scale patch pyramid

This approach creates a dense per-pixel embedding space where each location is represented by its learned local neighborhood structure across scales.

## Sample Output

The visualization below shows input patches (left) alongside their reconstructions (right) at multiple pyramid levels:

![Sample Output](docs/sample_output.png)

## Features

- **Per-pixel latent embeddings**: Each pixel gets its own latent vector encoding its local context
- **Multi-scale awareness**: Patch pyramid captures information at multiple receptive field sizes
- **VAE framework**: Probabilistic latent space with KL divergence regularization
- **U-Net architecture**: Skip connections preserve spatial detail during encoding/decoding

## Architecture

- **Encoder**: U-Net style with skip connections (16→32→64→128 channels)
- **Latent space**: Configurable dimension (default: 16) with VAE reparameterization
- **Decoder**: MLP-based per-pixel patch reconstruction
- **Output**: Multi-channel patch pyramid per pixel

## Requirements

```
torch
torchvision
PIL
```

## Usage

```bash
python patchwise-vae.py
```

Place training images in the `images/` directory. The model will train on 512×512 images and save debug visualizations to `debug/`.

## Configuration

Key parameters in the model:
- `latent_dim`: Size of the latent embedding per pixel (default: 16. **CAN BE INCREASED TO YOUR USE-CASE**)
- `patch_dim`: Base patch size (default: 8. **INCREASING EATS YOUR VRAM**)
- `pyramid_levels`: Number of pyramid scales (default: 3. **THIS EATS YOUR VRAM**)
