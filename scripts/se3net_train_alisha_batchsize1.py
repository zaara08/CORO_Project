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
    return transform(image).permute(0,1,2).numpy()  # Move channels to the last dimension and convert to numpy array



# Testing the dataset loader 
dataset = EpisodeDataset("/home/shashank/Documents/UniBonn/Sem4/alisha/Hind4Sight/Datasets/freiburg_real_poking/threeblocks/threeblocks/", device='cuda')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)



# Load a batch of episodes and print the shapes of the images and actions
batch = next(iter(dataloader))

model = SE3Net(3,4)
model.load_state_dict(torch.load("/home/shashank/Documents/UniBonn/Sem4/alisha/Hind4Sight/se3net/se3net_model.pth"))

# Training the model
# Define the loss function and optimizer
criterion = ImageReconstructionLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Training loop and visulaize the results and loss using tensorboard
writer = SummaryWriter()

num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(tqdm(dataloader)):
        # Get the inputs and labels
        x = data['rgb_1']
        x2 = data['rgb_2']
        u = data['action'].float()
        # Forward pass
        poses_new, x_new = model(x, u)
        # Compute the loss
        loss = criterion(x_new, x2)
        # Zero the gradients, backward pass, update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # Print the loss every 10 iterations
        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
        # Log the loss to tensorboard
        writer.add_scalar('training loss', loss.item(), epoch * len(dataloader) + i)
    # Visualize the results
    plot_image_from_se3_output(x_new)
    plot_image_from_se3_input_output_pair(x, x_new)
    plot_image_from_se3_input_output_gt(x, x2, x_new)
    # Save the model
    torch.save(model.state_dict(), 'se3net_model.pth')
    # Close the tensorboard writer
    writer.close()

# Save the model
torch.save(model.state_dict(), 'se3net_model.pth')
# Close the tensorboard writer
writer.close()
# End of script
