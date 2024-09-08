import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(file_path):
    image = Image.open(file_path).convert('L')  # Convert to grayscale
    return np.array(image)

def normalize_image(image, min_val=0, max_val=1):
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = image * (max_val - min_val) + min_val
    return image
#

def resize_image(image, target_size=(128, 128)):
    image = Image.fromarray(image)
    image = image.resize(target_size, Image.BILINEAR)
    return np.array(image)

def convert_to_tensor(image):
    image = np.expand_dims(image, axis=0)  # Add channel dimension
    tensor = torch.from_numpy(image).float()
    return tensor

class BrainDataset(Dataset):
    def __init__(self, image_files):
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = load_image(self.image_files[idx])
        img = normalize_image(img)
        img = resize_image(img)
        tensor = convert_to_tensor(img)
        return tensor.to(device)



class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=1*128*128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x).view(-1, 1, 128, 128)

class Discriminator(nn.Module):
    def __init__(self, img_dim=1*128*128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x.view(-1, 1*128*128))
    

    

def train_GAN(generator, discriminator, dataloader, epochs=100):
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(epochs):
        for real in dataloader:
            batch_size = real.size(0)
            real = real.view(batch_size, -1)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Train Discriminator
            d_optimizer.zero_grad()
            output = discriminator(real)
            real_loss = criterion(output, real_labels)

            noise = torch.randn(batch_size, 100)
            fake = generator(noise)
            output = discriminator(fake.detach())
            fake_loss = criterion(output, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            output = discriminator(fake)
            g_loss = criterion(output, real_labels)
            g_loss.backward()
            g_optimizer.step()

        print(f'Epoch [{epoch}/{epochs}] | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}')
        
    with torch.no_grad():
        fake_images = fake.reshape(-1, 1, 128, 128)
        fake_images = fake_images * 0.5 + 0.5  # Unnormalize from [-1, 1] to [0, 1]
        grid = torch.cat([fake_images[i] for i in range(3)], dim=1)
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        plt.savefig('brains.png', bbox_inches='tight', pad_inches=0)
        plt.show()

image_files = glob.glob('keras_png_slices_data\keras_png_slices_data\keras_png_slices_seg_train\*.png')
#dataset = BrainDataset(image_files[:100])
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#generator = Generator().to(device)
#discriminator = Discriminator().to(device)

#train_GAN(generator, discriminator, dataloader, epochs=10)
print(len(image_files))


