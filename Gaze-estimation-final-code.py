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


# Dataset class

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

if __name__ == "__main__":
   
   transform = transforms.Compose([
   transforms.Resize((64, 64)), #testirana velicina slika jos 32x32, 16x16 
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 3 kanala
   ])

   root_dir = 'C:/Users/andre/Desktop/Lukas/Lukas'

   train_dataset = EyeGazeDataset(txt_file=os.path.join(root_dir, 'train.txt'), root_dir=root_dir, transform=transform)
   val_dataset = EyeGazeDataset(txt_file=os.path.join(root_dir, 'validation.txt'), root_dir=root_dir, transform=transform)
   test_dataset = EyeGazeDataset(txt_file=os.path.join(root_dir, 'test.txt'), root_dir=root_dir, transform=transform)

   batch_size = 64

   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
   val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
   test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


   # Uzimamo jedan batch
   # dataiter = iter(train_loader) #vadimo jedan batch
   # images, labels = next(dataiter)

   
   # left_images = images[:, 0:3, :, :]   # [Batch, 3, H, W]
   # right_images = images[:, 3:6, :, :]  # [Batch, 3, H, W]

   # num_samples = 4  # ili koliko već 

   # # Kreiraj veliku figuru
   # fig, axs = plt.subplots(num_samples, 2, figsize=(8, num_samples * 4))

   # for i in range(num_samples):

   #    # Right Eye
   #    right_img_np = right_images[i].numpy() * 0.5 + 0.5  # unnormalize
   #    right_img_np = np.transpose(right_img_np, (1, 2, 0))
   #    axs[i, 0].imshow(right_img_np)
   #    axs[i, 0].set_title('Right Eye')
   #    axs[i, 0].axis('off')      

   #    # Left Eye
   #    left_img_np = left_images[i].numpy() * 0.5 + 0.5  # unnormalize
   #    left_img_np = np.transpose(left_img_np, (1, 2, 0))
   #    axs[i, 1].imshow(left_img_np)
   #    axs[i, 1].set_title('Left Eye')
   #    axs[i, 1].axis('off')




   # plt.tight_layout()
   #plt.show()

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
      
   
   mojModel=GazeNet()   
   total_params = sum(p.numel() for p in mojModel.parameters())
   trainable_params = sum(p.numel() for p in mojModel.parameters() if p.requires_grad)

   # print(f"Ukupan broj parametara: {total_params:,}")
   # print(f"Broj parametara koji se treniraju: {trainable_params:,}")


   loss_fn = torch.nn.L1Loss() #apsolutna pogreska

   if torch.cuda.is_available():
      my_device = torch.device('cuda')
      mojModel.to(my_device)
   else:
      my_device = torch.device('cpu')
   print(f'DEVICE: {my_device}')

   optimizer = torch.optim.Adam(mojModel.parameters(), lr=0.0001) #learning rate 10^-4.
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

   def train_one_epoch(epoch_index, model, optimizer, loss_fn, train_loader, device, tb_writer):
      running_loss = 0.0
      last_loss = 0.0

      # Postavljanje modela u trenirajući mod
      model.train()

      for i, data in enumerate(train_loader):
         # Dohvaćanje ulaza i labela iz batcha
         inputs, labels = data
         inputs, labels = inputs.to(device), labels.to(device)

         # Resetiranje gradijenata
         optimizer.zero_grad()

         # Forward pass
         outputs = model(inputs)

         # Izračun gubitka
         loss = loss_fn(outputs, labels)

         # Backward pass i optimizacija
         loss.backward()
         optimizer.step()

         # Akumulacija gubitka
         running_loss += loss.item()
         if i % 100 == 99:  # Izvještavanje svakih 100 batchova
            last_loss = running_loss / 100 #prosjecan loss zadnjih 100 batches
            print(f'Batch {i + 1}: Average Loss: {last_loss}')
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss kroz batcheve za svaku epohu', last_loss, tb_x)
            running_loss = 0.0

      return last_loss


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


# Glavni dio za treniranje i evaluaciju

   timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
   writer = SummaryWriter(f'runs_GAZE/gaze_trainer_{timestamp}')

   models_dir = os.path.join('models', f'models_{timestamp}')
   os.makedirs(models_dir, exist_ok=True)


   epoch_number = 0
   
   EPOCHS = 100
   best_vloss = float('inf')

   for epoch in range(EPOCHS):


      scheduler.step() #lr


      print(f'EPOCH {epoch_number + 1}:')

      # Treniranje modela za jednu epohu
      avg_loss = train_one_epoch(epoch_number, mojModel, optimizer, loss_fn, train_loader, my_device, writer)

      # Evaluacija modela na validation setu
      avg_vloss = evaluate_model(mojModel, loss_fn, val_loader, my_device)

      print(f'Training Loss: {avg_loss:.4f}, Validation Loss: {avg_vloss:.4f}')

      # Logiranje gubitka za treniranje i validaciju
      writer.add_scalars('Training vs. Validation Loss',
                        {'Training': avg_loss, 'Validation': avg_vloss},
                        epoch_number + 1)
      writer.flush()

      # Spremanje najboljeg modela
      if avg_vloss < best_vloss:
         best_vloss = avg_vloss
         model_path = os.path.join(models_dir, f'model_{timestamp}_epoch_{epoch_number + 1}.pth')
         torch.save(mojModel.state_dict(), model_path)

      epoch_number += 1


   #Evaluacija na test setu
   print('Evaluating on test set...')
   test_loss = evaluate_model(mojModel, loss_fn, test_loader, my_device)
   print(f'Test Loss: {test_loss:.4f}')

   # model_path="./models/models_2025-06-09_12-50/model_2025-06-09_12-50_epoch_1.pth"
   # print(f"TESTIRAM {model_path}")
   # mojModel.load_state_dict(torch.load(model_path))
   # mojModel.eval()
   # test_loss = evaluate_model(mojModel, loss_fn, test_loader, my_device)
   # print(f'Test Loss: {test_loss:.4f}')


   # def yaw_pitch_to_vec(yaw, pitch, length=20):
   #    dx = length * np.cos(pitch) * np.sin(yaw)
   #    dy = -length * np.sin(pitch)
   #    return dx, dy

   # # Učitaj najbolji model
   # mojModel.load_state_dict(torch.load(model_path))
   # mojModel.eval()

   # # Učitaj batch slika (npr. batch_size=8)
   # dataiter = iter(test_loader)
   # images, labels = next(dataiter)

   # # Ako koristiš GPU
   # images = images.to(my_device)

   # with torch.no_grad():
   #    preds = mojModel(images).cpu().numpy()

   # # Prikaži npr. 4 slike s predikcijama
   # num_samples = 15
   # for i in range(num_samples):
   #    # Priprema lijevog i desnog oka ([6, H, W], 3 kanala po oku)
   #    left_img = images[i, :3].cpu().numpy() * 0.5 + 0.5
   #    left_img = np.transpose(left_img, (1, 2, 0))
   #    right_img = images[i, 3:6].cpu().numpy() * 0.5 + 0.5
   #    right_img = np.transpose(right_img, (1, 2, 0))

   #    # Centar slike
   #    h, w = left_img.shape[:2]
   #    x0, y0 = w // 2, h // 2

   #    # Predikcija modela (crvena strelica)
   #    yaw_pred, pitch_pred = preds[i]
   #    dx_pred, dy_pred = yaw_pitch_to_vec(yaw_pred, pitch_pred, length=30)

   #    # Stvarna vrijednost (zelena strelica)
   #    yaw_true, pitch_true = labels[i].cpu().numpy()
   #    dx_true, dy_true = yaw_pitch_to_vec(yaw_true, pitch_true, length=30)

   #    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

   #    # Lijevo oko
   #    axs[1].imshow(left_img)
   #    axs[1].arrow(x0, y0, dx_pred, dy_pred, color='red', head_width=3, label='Predikcija')
   #    axs[1].arrow(x0, y0, dx_true, dy_true, color='green', head_width=3, label='Tocna vrijednost')
   #    axs[1].set_title('Lijevo oko')
   #    axs[1].axis('off')
   #    axs[1].legend(loc='upper right')

   #    # Desno oko
   #    axs[0].imshow(right_img)
   #    axs[0].arrow(x0, y0, dx_pred, dy_pred, color='red', head_width=3, label='Predikcija')
   #    axs[0].arrow(x0, y0, dx_true, dy_true, color='green', head_width=3, label='Tocna vrijednost')
   #    axs[0].set_title('Desno oko')
   #    axs[0].axis('off')
   #    axs[0].legend(loc='upper right')

   #    plt.suptitle(f"Predikcija: yaw={yaw_pred:.2f}, pitch={pitch_pred:.2f}\nTocno: yaw={yaw_true:.2f}, pitch={pitch_true:.2f}")
   #    plt.tight_layout()
   #    plt.show()

