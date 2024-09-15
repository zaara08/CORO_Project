import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from util.plot_utils import plot_image_from_se3_output, plot_image_from_se3_input_output_pair, plot_image_from_se3_input_output_gt
from models.se3net import SE3Net
from util.loss import ImageReconstructionLoss
from util.dataloader import EpisodeDataset
from torchvision import transforms

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size as needed
        transforms.ToTensor(),  # Convert to tensor
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).permute(0,1,2).numpy() 




dataset = EpisodeDataset("/home/Documents/UniBonn/Sem4/alisha/Hind4Sight/Datasets/freiburg_real_poking/threeblocks/threeblocks/", device='cuda')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)




batch = next(iter(dataloader))

model = SE3Net(3,4)
model = model.cuda()

criterion = ImageReconstructionLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000001)


writer = SummaryWriter()

num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(tqdm(dataloader)):

        x = data['rgb_1']
        x_depth = data['depth_1']
       
        x2 = data['rgb_2']
        x2_depth = data['depth_2']
        u = data['action'].float()
       
        poses_new, x_new = model(x, u)
        _, x_depth_new = model(x_depth, u)

        loss = criterion(x_new, x2) 
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            # # print both the losses
            # print("Image loss: ", criterion(x_new, x2).item())
            # print("Depth loss: ", criterion(x_depth_new, x2_depth).item())
            # print all the losses in one line
            # print('[%d, %5d] loss: %.3f, Image loss: %.3f, Depth loss: %.3f' % (epoch + 1, i + 1, running_loss / 10, criterion(x_new, x2).item(), criterion(x_depth_new, x2_depth).item()))
            running_loss = 0.0

        writer.add_scalar('training loss', loss.item(), epoch * len(dataloader) + i)

    torch.save(model.state_dict(), 'se3net_model.pth')

    writer.close()

torch.save(model.state_dict(), 'se3net_model.pth')
writer.close()

