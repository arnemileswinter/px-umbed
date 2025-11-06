import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

# Custom dataset for unlabeled images
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.glob('*.png')) + \
                          list(self.image_dir.glob('*.jpg')) + \
                          list(self.image_dir.glob('*.jpeg'))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
    
class PixelWisePatchPyramidVAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=8, out_channels=9, patch_dim=8):
        super(PixelWisePatchPyramidVAE, self).__init__()
        self.patch_dim = patch_dim
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        
        # Encoder with strided convolutions for downsampling
        self.unet_conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.unet_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.unet_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.unet_conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.unet_bottleneck = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Decoder with strided transposed convolutions for upsampling
        # UNet: upsample first, then concatenate with skip, then process
        self.unet_deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.unet_conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        
        self.unet_deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.unet_conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        
        self.unet_deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.unet_conv7 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)

        # VAE: separate heads for mean and log variance
        self.patch_head_mu = nn.Conv2d(16, latent_dim, kernel_size=3, padding=1)
        self.patch_head_logvar = nn.Conv2d(16, latent_dim, kernel_size=3, padding=1)
        
        # Reconstruct patches: decode each pixel's latent vector into an 8x8 patch
        # This is an MLP that works on individual latent vectors
        self.reconstruction_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, patch_dim * patch_dim)
        )
        
        # Add channel features with 1x1 conv
        self.channel_head_1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.channel_head_2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.channel_head_3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.channel_head_4 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x, batch_indices):
        batch_size, _, height, width = x.shape
        
        # Encoder with skip connections
        x1 = F.relu(self.unet_conv1(x))
        x2 = F.relu(self.unet_conv2(x1))
        x3 = F.relu(self.unet_conv3(x2))
        x4 = F.relu(self.unet_conv4(x3))
        x = F.relu(self.unet_bottleneck(x4))
        
        # Decoder with UNet-style concatenation: upsample -> concat -> process
        x = self.unet_deconv1(x)
        x = F.relu(self.unet_conv5(torch.cat([x, x3], dim=1)))
        
        x = self.unet_deconv2(x)
        x = F.relu(self.unet_conv6(torch.cat([x, x2], dim=1)))
        
        x = self.unet_deconv3(x)
        x = F.relu(self.unet_conv7(torch.cat([x, x1], dim=1)))

        # VAE: compute mean and log variance
        mu = self.patch_head_mu(x)
        logvar = self.patch_head_logvar(x)

        # Reshape from [batch, latent_dim, height, width] to [batch*height*width, latent_dim]
        mu = mu.permute(0, 2, 3, 1).contiguous().view(batch_size * height * width, self.latent_dim)
        logvar = logvar.permute(0, 2, 3, 1).contiguous().view(batch_size * height * width, self.latent_dim)
        
        # Select only the patches we need
        mu_selected = mu[batch_indices]
        logvar_selected = logvar[batch_indices]
        
        # Reparameterization trick: sample z = mu + std * epsilon
        std = torch.exp(0.5 * logvar_selected)
        eps = torch.randn_like(std)
        z = mu_selected + eps * std
        
        # We now have the per-pixel latent bottleneck with sampled z!
        # Compute KL divergence for selected patches
        kl_div = -0.5 * torch.sum(1 + logvar_selected - mu_selected.pow(2) - logvar_selected.exp(), dim=1).mean()

        # per pixel pyramid decoder:
        # Decode each latent vector into a patch
        x = self.reconstruction_head(z)  # [num_selected, patch_dim*patch_dim]
        # Reshape to [num_selected, 1, patch_dim, patch_dim]
        num_selected = z.shape[0]
        x = x.view(num_selected, 1, self.patch_dim, self.patch_dim)

        # Upscale for reconstruction
        x = F.relu(self.channel_head_1(x))
        x = F.relu(self.channel_head_2(x))
        x = F.relu(self.channel_head_3(x))
        x = torch.sigmoid(self.channel_head_4(x))
        
        return x, kl_div

# DataLoader for 512x512 images
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    image_dataset = ImageDataset(
        image_dir='images',
        transform=transform
    )

    dataloader = DataLoader(
        image_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    patch_dim = 8
    pyramid_levels = 3
    latent_dim = 16
    model = PixelWisePatchPyramidVAE(latent_dim=latent_dim, patch_dim=patch_dim).to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    patch_batch_size = 64

    for images in dataloader:
        images = images.to(device)
        
        # Pad images for patch extraction
        pad_size = patch_dim // 2
        padded_images = F.pad(images, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        # Pad images with zeros
        current_patch_dim = patch_dim
        pad_size = patch_dim // 2
        padded = F.pad(images, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        # Extract patch around each pixel using unfold with stride=1
        # unfold gives us 513x513 patches from 520x520, but we only want 512x512
        patches = padded.unfold(2, patch_dim, 1).unfold(3, patch_dim, 1)
        # Crop to keep only patches for the original 512x512 pixels
        patches = patches[:, :, :512, :512, :, :]  # [batch, channels, 512, 512, patch_h, patch_w]
        # Permute to get [batch, 512, 512, channels, patch_h, patch_w]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        # Reshape to [batch*512*512, channels, patch_h, patch_w]
        patch_batch, height, width, channels, patch_h, patch_w = patches.shape
        patches = patches.view(patch_batch * height * width, channels, patch_h, patch_w)

        for level in range(1, pyramid_levels):
            # create pyramid of patches - extract another sliding window of patch_dim * 2, pixel-wise,
            # then interpolate down to patch_dim and stack with original patches, such that
            # we get a tensor of shape [num_patches, channels, patch_dim, patch_dim]
            # pad again with pad_size * 2
            current_patch_dim *= 2
            pad_size = current_patch_dim // 2
            padded_pyramid = F.pad(images, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
            pyramid_patch = padded_pyramid.unfold(2, current_patch_dim, 1).unfold(3, current_patch_dim, 1)
            # Crop to keep only patches for the original 512x512 pixels
            pyramid_patch = pyramid_patch[:, :, :512, :512, :, :]
            # Shape after unfold: [batch, channels, 512, 512, patch_h, patch_w]
            pyramid_patch = pyramid_patch.permute(0, 2, 3, 1, 4, 5).contiguous()
            # Reshape to [batch*512*512, channels, patch_h, patch_w]
            p_batch, p_height, p_width, p_channels, p_patch_h, p_patch_w = pyramid_patch.shape
            pyramid_patch = pyramid_patch.view(p_batch * p_height * p_width, p_channels, p_patch_h, p_patch_w)
            pyramid_patch = F.interpolate(pyramid_patch, size=(patch_dim, patch_dim))
            # Concatenate with original patches along channel dimension, yielding [num_patches, channels + in_channels, patch_dim, patch_dim]
            patches = torch.cat((patches, pyramid_patch), dim=1)

        print("Total patches shape with pyramid levels:", patches.shape)
        print("Number of patches:", patches.shape[0])
        print("Number of patch batches:", patches.shape[0] // patch_batch_size)
        
        # Static indices for intentional overfitting (no randomization)
        indices = torch.randperm(patches.shape[0])
        
        for patch_batch in range(patches.shape[0] // patch_batch_size):
            # Use shuffled indices to grab random patches
            batch_indices = indices[patch_batch * patch_batch_size:(patch_batch + 1) * patch_batch_size]
            patch_batch = patches[batch_indices] # grab random per-pixel pyramid-patches
            
            # Get model outputs and KL divergence
            # model is fed entire image, batch_indices will be the entirety of its pixel indices in inference.
            # currently we clip to only the pixel indices for patches we want to train on.
            outputs, kl_div = model(images, batch_indices)
            
            optimizer.zero_grad()

            # Total loss = reconstruction loss + KL divergence
            recon_loss = criterion(outputs, patch_batch)

            kl_weight = 0.0001
            loss = recon_loss + kl_weight * kl_div
            loss.backward()
            optimizer.step()
            print(f'Patch-Batch {patch_batch}, Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_div.item():.4f}')

            # debug visualize the patch batch inputs and outputs
            if (patch_batch+1) % 100 == 0:
                all_comparisons = []
                
                for i in range(patch_batch_size):
                    input_patch = patch_batch[i].detach().cpu()
                    output_patch = outputs[i].detach().cpu()
                    
                    # Create a 2-column layout: inputs on left, outputs on right
                    # Stack pyramid levels vertically
                    input_levels = []
                    output_levels = []
                    
                    for l in range(pyramid_levels):
                        # Visualize each level of the pyramid
                        start_channel = l * 3
                        end_channel = start_channel + 3
                        input_level = input_patch[start_channel:end_channel, :, :]
                        output_level = output_patch[start_channel:end_channel, :, :]
                        input_levels.append(input_level)
                        output_levels.append(output_level)
                    
                    # Stack levels vertically (along height dimension)
                    input_column = torch.cat(input_levels, dim=1)
                    output_column = torch.cat(output_levels, dim=1)
                    
                    # Concatenate horizontally to create 2-column layout
                    comparison = torch.cat([input_column, output_column], dim=2)
                    all_comparisons.append(comparison)
                
                # Stack all patch comparisons vertically into one large image
                final_image = torch.cat(all_comparisons, dim=1)
                
                # Save the combined image
                transforms.ToPILImage()(final_image).save(f'debug/batch_{patch_batch}_all_patches.png')

        print('Patches shape:', patches.shape)
