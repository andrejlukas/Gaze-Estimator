import math
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

import torchvision.models as models
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class EyeGazeDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        self.samples = []
        self.root_dir = root_dir
        self.transform = transform

        with open(txt_file, 'r') as f:
            for line in f:
                path, x, y, z = line.strip().split()
                self.samples.append((path, float(x), float(y), float(z)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, x, y, z = self.samples[idx]

        parts = img_path.split('/')
        rec_folder = parts[0]
        #head_folder = parts[1]
        eye_folder = parts[2]
        img_name = parts[3]

        left_img_path = os.path.join(self.root_dir, rec_folder, "eye", eye_folder, "left_" + img_name)
        right_img_path = os.path.join(self.root_dir, rec_folder, "eye", eye_folder, "right_" + img_name)

        # Učitavanje slika
        left_img = Image.open(left_img_path)
        right_img = Image.open(right_img_path)

        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        # Stackamo slike zajedno
        image_pair = torch.cat([left_img, right_img], dim=0)  # (6, H, W)

        
        # x, y, z = 0.625781069236190, -0.385936719702466, -0.677828076853498
        yaw = math.atan2(x,-z)
        pitch = math.asin(y)

        yaw_deg = math.degrees(yaw) 
        pitch_deg = math.degrees(pitch)
        # print(f"yaw: {yaw}== {yaw_deg} stupnjeva \npitch: {pitch}== {pitch_deg} stupenjva")
        label = torch.tensor([yaw, pitch], dtype=torch.float32)

        return image_pair, label
   
class GazeNet(nn.Module):
    def __init__(self):
        super(GazeNet, self).__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Uzimamo backbone bez zadnjeg FC sloja
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.fc = nn.Linear(512, 2)  # 512 izlaza iz ResNet backbonea

    def forward(self, image_pair):

        features = self.backbone(image_pair)  # output shape: (batch_size, 512, 1, 1)
        # Flattenanje feature map-a (maknemo višak dimenzija)
        features = features.view(features.size(0), -1)  # (batch_size, 512)

        # Linearni sloj za predikciju yaw i pitch
        output = self.fc(features)  # (batch_size, 2)

        return output
    
def evaluate_model(model, loss_fn, my_loader, device):
    running_vloss = 0.0

    #Postavljanje modela u evaluacijski mod
    model.eval()

    #Isključivanje gradijenata za evaluaciju
    with torch.no_grad():
        for i, vdata in enumerate(my_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)

            #Forward pass
            voutputs = model(vinputs)

            #Izračun gubitka
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.item()

    avg_vloss = running_vloss / (i + 1)
    return avg_vloss
   
if __name__ == "__main__":

    mojModel=GazeNet()

    if torch.cuda.is_available():
        my_device = torch.device('cuda')
        mojModel.to(my_device)
    else:
        my_device = torch.device('cpu')
    print(f'DEVICE: {my_device}')

    transform = transforms.Compose([
    transforms.Resize((64, 64)), #testirana velicina slika jos 32x32, 16x16 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 3 kanala
    ])

    root_dir = 'C:/Users/andre/Desktop/Lukas/Lukas'

    test_dataset = EyeGazeDataset(txt_file=os.path.join(root_dir, 'test.txt'), root_dir=root_dir, transform=transform)

    batch_size = 64

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    model_path="./models/models_2026-03-05_12-13/model_2026-03-05_12-13_epoch_36.pth"
    
    print(f"TESTIRAM {model_path}")

    loss_fn = torch.nn.L1Loss()

    mojModel.load_state_dict(torch.load(model_path))
    mojModel.eval()
    test_loss = evaluate_model(mojModel, loss_fn, test_loader, my_device)
    print(f'Test Loss: {test_loss:.4f}')


    def yaw_pitch_to_vec(yaw, pitch, length=20):
        dx = length * np.cos(pitch) * np.sin(yaw)
        dy = -length * np.sin(pitch)
        return dx, dy

    # Učitaj najbolji model
    mojModel.load_state_dict(torch.load(model_path))
    mojModel.eval()

    # Učitaj batch slika (npr. batch_size=8)
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # Ako koristiš GPU
    images = images.to(my_device)

    with torch.no_grad():
        preds = mojModel(images).cpu().numpy()

    # Prikaži npr. 4 slike s predikcijama
    num_samples = 4
    for i in range(num_samples):
        # Priprema lijevog i desnog oka ([6, H, W], 3 kanala po oku)
        left_img = images[i, :3].cpu().numpy() * 0.5 + 0.5
        left_img = np.transpose(left_img, (1, 2, 0))
        right_img = images[i, 3:6].cpu().numpy() * 0.5 + 0.5
        right_img = np.transpose(right_img, (1, 2, 0))

        # Centar slike
        h, w = left_img.shape[:2]
        x0, y0 = w // 2, h // 2

        # Predikcija modela (crvena strelica)
        yaw_pred, pitch_pred = preds[i]
        dx_pred, dy_pred = yaw_pitch_to_vec(yaw_pred, pitch_pred, length=30)

        # Stvarna vrijednost (zelena strelica)
        yaw_true, pitch_true = labels[i].cpu().numpy()
        dx_true, dy_true = yaw_pitch_to_vec(yaw_true, pitch_true, length=30)

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        # Lijevo oko
        axs[1].imshow(left_img)
        axs[1].arrow(x0, y0, dx_pred, dy_pred, color='red', head_width=3, label='Predikcija')
        axs[1].arrow(x0, y0, dx_true, dy_true, color='green', head_width=3, label='Tocna vrijednost')
        axs[1].set_title('Lijevo oko')
        axs[1].axis('off')
        axs[1].legend(loc='upper right')

        # Desno oko
        axs[0].imshow(right_img)
        axs[0].arrow(x0, y0, dx_pred, dy_pred, color='red', head_width=3, label='Predikcija')
        axs[0].arrow(x0, y0, dx_true, dy_true, color='green', head_width=3, label='Tocna vrijednost')
        axs[0].set_title('Desno oko')
        axs[0].axis('off')
        axs[0].legend(loc='upper right')

        plt.suptitle(f"Predikcija: yaw={yaw_pred:.2f}, pitch={pitch_pred:.2f}\nTocno: yaw={yaw_true:.2f}, pitch={pitch_true:.2f}")
        plt.tight_layout()
        plt.show()
