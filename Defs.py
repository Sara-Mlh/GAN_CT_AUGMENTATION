# Function to save checkpoints
import os 
import SimpleITK as sitk
from torchvision.transforms import transforms
import torch 
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import io
from PIL import Image
import base64
import glob
import torch.nn as nn
from io import BytesIO


def save_checkpoint(epoch, G, D, G_opt, D_opt, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'G_opt_state_dict': G_opt.state_dict(),
        'D_opt_state_dict': D_opt.state_dict(),
    }, checkpoint_path)

# Function to load checkpoints
def load_checkpoint(checkpoint_path, G, D, G_opt, D_opt):
    checkpoint = torch.load(checkpoint_path)
    G.load_state_dict(checkpoint['G_state_dict'])
    D.load_state_dict(checkpoint['D_state_dict'])
    G_opt.load_state_dict(checkpoint['G_opt_state_dict'])
    D_opt.load_state_dict(checkpoint['D_opt_state_dict'])
    return checkpoint['epoch']

class CTScanDataset(Dataset):
    def __init__(self, file_paths, resize_size=(128,128)):
        self.file_paths = file_paths
        self.resize = transforms.Resize(resize_size)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            image = sitk.ReadImage(file_path)
            image_array = sitk.GetArrayFromImage(image)
            #logging.info(f"Successfully loaded image from {file_path}")

            # Convert to a tensor
            image_tensor = torch.tensor(image_array, dtype=torch.float32)

            # normalization 
            min_val = torch.min(image_tensor)
            max_val = torch.max(image_tensor)
            normalized_image = (image_tensor - min_val) / (max_val - min_val) * 2 - 1
            #resize
            image_tensor = self.resize(normalized_image.unsqueeze(0)).squeeze(0)

            return image_tensor

        except Exception as e:
            #logging.error(f"Error reading {file_path}: {e}")
            new_idx = (idx + 1) % len(self.file_paths)
            return self.__getitem__(new_idx)
        
def dicom_paths(path):
   file_paths = glob.glob(os.path.join(path, '**', '*.dcm'), recursive=True)
   return file_paths



def show_tensor_images(tensor_img, num_images=16):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(tensor_img[i].squeeze(), cmap='gray')
        ax.axis('off')

    plt.tight_layout()

    # Save to bytesIO buffer and encode it as base64
    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    buf.seek(0)
    
    # Encode the image as base64
    img_base64 = base64.b64encode(buf.getvalue()).decode('latin1')
    
    plt.close(fig)
    
    return img_base64  



def show_images(tensor_img, num_images=16):
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(tensor_img[i].squeeze(), cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #nn.ConvTranspose2d(noise_dim, 1024, 4, 1, 0, bias=False),
            #nn.BatchNorm2d(1024),
            #nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(1024),
            #nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    


def show_generated_images(epoch, generator, num_images=16, noise_dim=100):
    device ="cuda"
    noise = torch.randn(num_images, noise_dim, 1, 1, device=device)
    with torch.no_grad():
        fake_images = generator(noise).cpu()

    # Denormalize the images from [-1, 1] to [0, 1]
    fake_images = (fake_images + 1) / 2.0

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(fake_images[i].squeeze(), cmap='gray')
        ax.axis('off')

    plt.tight_layout()

    # Save to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    # Encode as base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    
    return img_str

