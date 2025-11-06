import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import models
from PIL import Image
import os
from pathlib import Path

# Perceptual Loss using pretrained VGG features
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Load pretrained VGG16 and extract feature layers
        vgg = models.vgg16(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:16]).eval()
        
        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, inputs, targets):
        # Handle multi-pyramid channel inputs (9 channels = 3 levels × 3 RGB)
        # Process each pyramid level separately
        num_channels = inputs.shape[1]
        channels_per_level = 3
        num_levels = num_channels // channels_per_level
        
        total_loss = 0
        for i in range(num_levels):
            start_ch = i * channels_per_level
            end_ch = start_ch + channels_per_level
            
            input_level = inputs[:, start_ch:end_ch, :, :]
            target_level = targets[:, start_ch:end_ch, :, :]

            # Upsample to minimum VGG input size if needed (8x8 -> 32x32)
            if input_level.shape[2] < 32 or input_level.shape[3] < 32:
                input_level = F.interpolate(input_level, size=(32, 32), mode='bilinear', align_corners=False)
                target_level = F.interpolate(target_level, size=(32, 32), mode='bilinear', align_corners=False)
            
            # Extract features
            input_features = self.feature_extractor(input_level)
            target_features = self.feature_extractor(target_level)
            
            # Compute MSE in feature space
            total_loss += F.mse_loss(input_features, target_features)
        
        return total_loss / num_levels

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
        self.unet_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.unet_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 512 -> 256
        self.unet_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 256 -> 128
        self.unet_conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 128 -> 64

        self.unet_bottleneck = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        # Decoder with strided transposed convolutions for upsampling
        self.unet_deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # 64 -> 128
        self.unet_deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 128 -> 256
        self.unet_deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 256 -> 512

        # VAE: separate heads for mean and log variance
        self.patch_head_mu = nn.Conv2d(64, latent_dim, kernel_size=3, padding=1)
        self.patch_head_logvar = nn.Conv2d(64, latent_dim, kernel_size=3, padding=1)
        
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
        self.channel_head_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.channel_head_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.channel_head_3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x, batch_indices):
        batch_size, _, height, width = x.shape
        
        # Encoder with skip connections
        x1 = F.relu(self.unet_conv1(x))  # [B, 64, 512, 512]
        x2 = F.relu(self.unet_conv2(x1))  # [B, 128, 256, 256]
        x3 = F.relu(self.unet_conv3(x2))  # [B, 256, 128, 128]
        x4 = F.relu(self.unet_conv4(x3))  # [B, 512, 64, 64]
        x = F.relu(self.unet_bottleneck(x4))  # Bottleneck
        # Decoder with skip connections
        x = F.relu(self.unet_deconv1(x) + x3)  # [B, 256, 128, 128]
        x = F.relu(self.unet_deconv2(x) + x2)  # [B, 128, 256, 256]
        x = F.relu(self.unet_deconv3(x) + x1)  # [B, 64, 512, 512]

        x = self.patch_head(x)

        # Reshape from [batch, latent_dim, height, width] to [batch*height*width, latent_dim]
        x = x.permute(0, 2, 3, 1).contiguous()  # [batch, height, width, latent_dim]
        x = x.view(batch_size * height * width, self.latent_dim)[batch_indices]  # [num_selected, latent_dim]
        # We now have the per-pixel latent bottleneck!

        # todo: we need KL divergence??

        # per pixel pyramid decoder:
        # Decode each latent vector into a patch
        x = self.reconstruction_head(x)  # [num_selected, patch_dim*patch_dim]
        # Reshape to [num_selected, 1, patch_dim, patch_dim]
        num_selected = x.shape[0]
        x = x.view(num_selected, 1, self.patch_dim, self.patch_dim)
        # Add channel features
        x = F.relu(self.channel_head_1(x))  # [batch*height*width, 32, patch_dim, patch_dim]
        x = F.relu(self.channel_head_2(x))  # [batch*height*width, 64, patch_dim, patch_dim]
        x = F.tanh(self.channel_head_3(x))  # [batch*height*width, out_channels, patch_dim, patch_dim]
        
        return x

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

    criterion = PerceptualLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    patch_batch_size = 128

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
        batch, height, width, channels, patch_h, patch_w = patches.shape
        patches = patches.view(batch * height * width, channels, patch_h, patch_w)

        for level in range(1, pyramid_levels):
            # create pyramid of patches - extract another sliding window of patch_dim * 2, pixel-wise, then interpolate down to patch_dim and stack with original patches, such that
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
        
        # Shuffle patch indices for randomized training
        indices = torch.randperm(patches.shape[0])
        
        for batch in range(patches.shape[0] // patch_batch_size):
            # Use shuffled indices to grab random patches
            batch_indices = indices[batch * patch_batch_size:(batch + 1) * patch_batch_size]
            patch_batch = patches[batch_indices]  # grab random per-pixel pyramid-patches
            
            # Apply consistent augmentations across all pyramid levels for each patch
            augmented_patches = []
            for i in range(patch_batch.shape[0]):
                # Randomly decide augmentation parameters for this patch
                do_hflip = torch.rand(1).item() > 0.5
                do_vflip = torch.rand(1).item() > 0.5
                rotation_k = torch.randint(0, 4, (1,)).item()  # 0=no rotation, 1=90°, 2=180°, 3=270°
                hue_factor = (torch.rand(1).item() - 0.5) * 0.6  # range [-0.3, 0.3]
                
                # Apply the same transformations to all pyramid levels
                levels = []
                for l in range(pyramid_levels):
                    start_channel = l * 3
                    end_channel = start_channel + 3
                    level = patch_batch[i, start_channel:end_channel, :, :]
                    
                    # Apply transformations
                    if do_hflip:
                        level = torch.flip(level, [2])  # flip width
                    if do_vflip:
                        level = torch.flip(level, [1])  # flip height
                    if rotation_k > 0:
                        level = torch.rot90(level, k=rotation_k, dims=[1, 2])
                    
                    # Apply hue shift in GPU using RGB to HSV conversion
                    if hue_factor != 0:
                        # Simple RGB-based hue shift approximation (stays on GPU)
                        # Shift color channels cyclically
                        r, g, b = level[0], level[1], level[2]
                        shift_amount = hue_factor * 2  # scale to [-0.6, 0.6]
                        
                        # Apply color shift
                        level = torch.stack([
                            torch.clamp(r + shift_amount * (g - b), 0, 1),
                            torch.clamp(g + shift_amount * (b - r), 0, 1),
                            torch.clamp(b + shift_amount * (r - g), 0, 1)
                        ], dim=0)
                    
                    levels.append(level)
                
                # Concatenate all levels back together
                aug_patch = torch.cat(levels, dim=0)
                augmented_patches.append(aug_patch)
            
            patch_batch = torch.stack(augmented_patches)
            
            # Get model outputs and select the same indices
            outputs = model(images, batch_indices)
            
            optimizer.zero_grad()
            loss = criterion(outputs, patch_batch)
            loss.backward()
            optimizer.step()
            print(f'Patch-Batch {batch}, Loss: {loss.item()}')

            # debug visualize the patch batch inputs and outputs
            if (batch+1) % 100 == 0:
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
                    input_column = torch.cat(input_levels, dim=1)  # [3, height*pyramid_levels, width]
                    output_column = torch.cat(output_levels, dim=1)  # [3, height*pyramid_levels, width]
                    
                    # Concatenate horizontally to create 2-column layout
                    comparison = torch.cat([input_column, output_column], dim=2)  # [3, height*pyramid_levels, width*2]
                    all_comparisons.append(comparison)
                
                # Stack all patch comparisons vertically into one large image
                final_image = torch.cat(all_comparisons, dim=1)  # [3, height*pyramid_levels*batch_size, width*2]
                
                # Save the combined image
                transforms.ToPILImage()(final_image).save(f'debug/batch_{batch}_all_patches.png')

        print('Patches shape:', patches.shape)
