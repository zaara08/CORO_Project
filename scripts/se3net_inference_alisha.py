import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from util.plot_utils import plot_image_from_se3_output, plot_image_from_se3_input_output_pair, plot_image_from_se3_input_output_gt
from models.se3net import SE3Net
from util.loss import ImageReconstructionLoss
from util.dataloader import EpisodeDataset


# Testing the dataset loader 
dataset = EpisodeDataset("/home/Documents/UniBonn/Sem4/alisha/Hind4Sight/Datasets/freiburg_real_poking/threeblocks/threeblocks/")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)


from matplotlib import pyplot as plt

# Load a batch of episodes and print the shapes of the images and actions
batch = next(iter(dataloader))
print(batch['depth_1'].shape, batch['depth_2'].shape, batch['flow'].shape, batch['rgb_1'].shape, batch['rgb_2'].shape)
print(batch['action'].shape, batch['action_ang'].shape, batch['crop_info'].shape)


model = SE3Net(3,4)
model.load_state_dict(torch.load("/home/Documents/UniBonn/Sem4/alisha/Hind4Sight/se3net/se3net_model.pth"))
x = batch['rgb_1']
x_depth = batch['depth_1'] # 1, 1, 224, 224
x_depth_rgb = x_depth.repeat(1, 3, 1, 1) # 1, 3, 224, 224
x2 = batch['rgb_2']
x2_depth = batch['depth_2']
x2_depth_rgb = x2_depth.repeat(1, 3, 1, 1)
u = batch['action'].float()

pt1 = batch['pt1'] # 1, 3, 224, 224
pt2 = batch['pt2']
print("pt1.shape, pt2.shape: ", pt1.shape, pt2.shape)
# Visualize the point cloud data using open3d
import open3d as o3d
# Open3D PCD shape: (N, 3)
pcd = o3d.geometry.PointCloud()
pt1 = pt1.squeeze().cpu().detach().numpy() # 3, 224, 224
pt2 = pt2.squeeze().cpu().detach().numpy() # 3, 224, 224

print("pt1.shape, pt2.shape: ", pt1.shape, pt2.shape)
# Convert to (224*224, 3)
pt1 = np.transpose(pt1, (1, 2, 0)).reshape(-1, 3)
pt2 = np.transpose(pt2, (1, 2, 0)).reshape(-1, 3)
print("pt1.shape, pt2.shape: ", pt1.shape, pt2.shape)

pt1_color = batch['rgb1'].squeeze().cpu().detach().numpy() # 3, 224, 224
pt2_color = batch['rgb2'].squeeze().cpu().detach().numpy() # 3, 224, 224
pt1_color = np.transpose(pt1_color, (1, 2, 0)).reshape(-1, 3)
pt2_color = np.transpose(pt2_color, (1, 2, 0)).reshape(-1, 3)
print("pt1_color.shape, pt2_color.shape: ", pt1_color.shape, pt2_color.shape)


# Visualize the point cloud data
pcd.points = o3d.utility.Vector3dVector(pt1)

# Add colors to the point cloud
pcd.colors = o3d.utility.Vector3dVector(pt1_color)

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd, axis])




# Print datatypes of x, u
print(x.dtype, u.dtype)

# Forward pass
poses_new, x_new = model(x, u)
_, x_new_depth = model(x_depth_rgb, u)
print(" x_new.shape, poses_new.shape ", x_new.shape, poses_new.shape)
plot_image_from_se3_output(x_new)
plot_image_from_se3_input_output_pair(x, x_new)
plot_image_from_se3_input_output_gt(x, x2, x_new)
plot_image_from_se3_input_output_gt(x_depth_rgb, x2_depth_rgb, x_new_depth)

# Check if the images have any nan or inf values, x, x2, x_new
if torch.isnan(x).any() or torch.isinf(x).any():
    raise ValueError("input_image contains NaN or Inf values.")
if torch.isnan(x_new).any() or torch.isinf(x_new).any():
    raise ValueError("predicted_image contains NaN or Inf values.")
if torch.isnan(x2).any() or torch.isinf(x2).any():
    raise ValueError("ground_truth_image contains NaN or Inf values.")


# Define the loss function
image_loss = ImageReconstructionLoss()
# Calculate the loss
image_loss_1 = image_loss(x_new, x2)
image_loss_2 = image_loss(x_new_depth, x2_depth_rgb)
print("Image Loss rgb: ", image_loss_1)
print("Image Loss depth: ", image_loss_2)

