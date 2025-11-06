import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

# Custom transform to add random noise
class AddRandomNoise:
    def __init__(self, noise_factor=0.05):
        self.noise_factor = noise_factor
    
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.noise_factor
        return torch.clamp(tensor + noise, 0., 1.)

# Simple downsampling block
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.conv(x))

# Simple upsampling block with skip connection
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(UpBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)
        self.conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x, skip):
        x = self.deconv(x)
        x = torch.cat([x, skip], dim=1)
        return self.relu(self.conv(x))

# Dataset that constructs patch pyramids for random pixels
class PatchPyramidDataset(Dataset):
    def __init__(self, image_dir, transform=None, patch_dim=16, pyramid_levels=3, patches_per_image=8, preload=True):
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.glob('*.png')) + \
                          list(self.image_dir.glob('*.jpg')) + \
                          list(self.image_dir.glob('*.jpeg'))
        self.transform = transform
        self.patch_dim = patch_dim
        self.pyramid_levels = pyramid_levels
        self.patches_per_image = patches_per_image
        self.preload = preload
        
        # Preload and cache all images in memory
        if self.preload:
            print(f'Preloading {len(self.image_paths)} images into memory...')
            self.images = []
            for img_path in self.image_paths:
                image = Image.open(img_path).convert('RGB')
                self.images.append(image)
            print('Preloading complete!')
        else:
            self.images = None
    
    def __len__(self):
        return len(self.image_paths) * self.patches_per_image
    
    def __getitem__(self, idx):
        # Map idx to image and patch within that image
        img_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image
        
        # Get image from cache or load from disk
        if self.preload:
            image = self.images[img_idx]
        else:
            img_path = self.image_paths[img_idx]
            image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # image is now a tensor [C, H, W]
        channels, height, width = image.shape
        
        # Calculate the maximum patch size needed for the highest pyramid level
        max_patch_dim = self.patch_dim * (2 ** (self.pyramid_levels - 1))
        half_max_patch = max_patch_dim // 2
        
        # Select a random pixel position that avoids edges
        # This ensures we can extract patches at all pyramid levels without padding
        py = torch.randint(half_max_patch, height - half_max_patch, (1,)).item()
        px = torch.randint(half_max_patch, width - half_max_patch, (1,)).item()
        pixel_idx = py * width + px
        
        # Build patch pyramid only for this pixel
        pyramid_patches = []
        
        for level in range(self.pyramid_levels):
            level_patch_dim = self.patch_dim * (2 ** level)
            half_patch = level_patch_dim // 2
            
            # Extract patch centered at (py, px) - no padding needed
            y_start = py - half_patch
            x_start = px - half_patch
            
            patch = image[:, y_start:y_start+level_patch_dim, x_start:x_start+level_patch_dim]
            
            # Resize to standard patch_dim
            if level > 0:
                patch = F.interpolate(patch.unsqueeze(0), size=(self.patch_dim, self.patch_dim), mode='bilinear', align_corners=False)
                patch = patch.squeeze(0)
            
            pyramid_patches.append(patch)
        
        # Concatenate along channel dimension
        pyramid_patch = torch.cat(pyramid_patches, dim=0)  # [pyramid_levels*channels, patch_dim, patch_dim]
        
        return image, pyramid_patch, pixel_idx
    
class PixelWisePatchPyramidVAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=8, out_channels=9, patch_dim=8):
        super(PixelWisePatchPyramidVAE, self).__init__()
        self.patch_dim = patch_dim
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        
        # Encoder with strided convolutions for downsampling
        self.down1 = DownBlock(in_channels, 16, stride=1)
        self.down2 = DownBlock(16, 32, stride=2)
        self.down3 = DownBlock(32, 64, stride=2)
        self.down4 = DownBlock(64, 128, stride=2)
        self.down5 = DownBlock(128, 256, stride=2)
        self.down6 = DownBlock(256, 512, stride=2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Decoder with strided transposed convolutions for upsampling
        self.up1 = UpBlock(512, 256, stride=2)
        self.up2 = UpBlock(256, 128, stride=2)
        self.up3 = UpBlock(128, 64, stride=2)
        self.up4 = UpBlock(64, 32, stride=2)
        self.up5 = UpBlock(32, 16, stride=2)

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
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        
        x = self.bottleneck(x6)
        
        # Decoder with UNet-style skip connections
        x = self.up1(x, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)

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

# DataLoader for 128x128 images
img_dim=128
transform = transforms.Compose([
    transforms.RandomCrop((img_dim, img_dim)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(hue=0.5),  # Random hue shift across full palette
    transforms.ToTensor(),
    AddRandomNoise(noise_factor=0.02),  # Add random noise after converting to tensor
])

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    patch_dim = 16
    pyramid_levels = 3
    patches_per_image = 10_000

    image_dataset = PatchPyramidDataset(
        image_dir='images',
        transform=transform,
        patch_dim=patch_dim,
        pyramid_levels=pyramid_levels,
        patches_per_image=patches_per_image
    )

    dataloader = DataLoader(
        image_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )

    latent_dim = 12
    model = PixelWisePatchPyramidVAE(latent_dim=latent_dim, patch_dim=patch_dim).to(device)

    kl_weight = 0.0001  # Reduced initial KL weight
    kl_anneal_rate = 0.0001  # Gradually increase KL weight
    max_kl_weight = 0.01  # Maximum KL weight
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    epochs = 100
    save_interval = 5  # Save model snapshot every 5 epochs

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0
        
        # KL annealing: gradually increase KL weight
        current_kl_weight = min(kl_weight + epoch * kl_anneal_rate, max_kl_weight)
        
        print(f'\nEpoch {epoch+1}/{epochs} (KL weight: {current_kl_weight:.6f})')
        
        for images, patches, patch_indices in dataloader:
            images = images.to(device)
            patches = patches.to(device)
            patch_indices = patch_indices.to(device)
            
            # Get model outputs and KL divergence
            outputs, kl_div = model(images, patch_indices)
            
            optimizer.zero_grad()

            # Total loss = reconstruction loss + KL divergence
            recon_loss = criterion(outputs, patches)

            loss = recon_loss + current_kl_weight * kl_div
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate loss for epoch statistics
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_div.item()
            num_batches += 1
            
            # Print dot for batch progress
            print('.', end='', flush=True)

            # debug visualize the patch batch inputs and outputs
            if num_batches % 100 == 0:
                batch_size = patches.shape[0]
                all_comparisons = []
                
                for i in range(min(batch_size, 64)):
                    input_patch = patches[i].detach().cpu()
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
                transforms.ToPILImage()(final_image).save(f'debug/epoch_{epoch}_batch_{num_batches}_all_patches.png')
        
                # Print epoch statistics
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_recon_loss = epoch_recon_loss / num_batches if num_batches > 0 else 0
        avg_kl_loss = epoch_kl_loss / num_batches if num_batches > 0 else 0
        print(f'\nEpoch {epoch+1} completed - Total Loss: {avg_epoch_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}')
        
        # Save model snapshot every few epochs
        if (epoch + 1) % save_interval == 0:
            snapshot_path = f'model_snapshot_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'recon_loss': avg_recon_loss,
                'kl_loss': avg_kl_loss,
            }, snapshot_path)
            print(f'Model snapshot saved to {snapshot_path}')
