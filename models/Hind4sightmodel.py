import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.nn import functional as F
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d
from torchvision import models
from torch.autograd import Function, Variable
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as f
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from copy import deepcopy as copy
import math
import sys
from pdb import set_trace as st 

sys.path.append('..')
from util.rot_utils import axisAngleToRotationMatrix_batched, rotationMatrixToAxisAngle_batched
from util.network_utils import apply

import cv2
import matplotlib.pyplot as plt
from pytorch3d.loss import chamfer_distance


class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

class BatchNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BatchNormConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm(x)
        return F.relu(x, inplace=True)

class BatchNormDeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BatchNormDeconv2d, self).__init__()
        self.deconv2d = UpsampleConvLayer(in_channels, out_channels, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        x = self.deconv2d(x)
        x = self.batch_norm(x)
        return F.relu(x, inplace=True)

class FCN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(FCN, self).__init__()
        self.deconv2d = UpsampleConvLayer(in_channels, out_channels, **kwargs)


    def forward(self, x):
        x = self.deconv2d(x)
        return F.relu(x, inplace=True)

class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x, inplace=True)
        return x


class PoseAndMaskEncoder(nn.Module):
    """Pose Nets
    """
    def __init__(self, k):
        #Implementation a mix of SE3 Nets and SE3 Pose Nets
        super(PoseAndMaskEncoder, self).__init__()
        self.k = k 
        # Encoder
        self.Conv1 = Conv2d(3, 8, bias=False, kernel_size=2, stride=1, padding=1)
        self.Pool1 = MaxPool2d(kernel_size=2)
        self.Conv2 = BatchNormConv2d(8, 16, bias=False, kernel_size=3, stride=1, padding=1)
        self.Pool2 = MaxPool2d(kernel_size=2)
        self.Conv3 = BatchNormConv2d(16, 32, bias=False, kernel_size=3, stride=1, padding=1)
        self.Pool3 = MaxPool2d(kernel_size=2)
        self.Conv4 = BatchNormConv2d(32, 64, bias=False, kernel_size=3, stride=1, padding=1)
        self.Pool4 = MaxPool2d(kernel_size=2)
        self.Conv5 = BatchNormConv2d(64, 128, bias=False, kernel_size=3, stride=1, padding=1)
        self.Pool5 = MaxPool2d(kernel_size=2)
        # Mask Decoder
        self.Deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2) #, output_padding=1)
        self.Deconv2 = BatchNormDeconv2d(64, 32, kernel_size=3, stride=1, upsample=2) #, output_padding=1)
        self.Deconv3 = BatchNormDeconv2d(32, 16, kernel_size=3, stride=1, upsample=2) #, output_padding=1)
        self.Deconv4 = BatchNormDeconv2d(16, 8, kernel_size=3, stride=1, upsample=2) #, output_padding=1)
        self.Deconv5 = FCN(8, self.k, kernel_size=3, stride=1, upsample=2) #, output_padding=1)
        # Pose Decoder
        self.Fc1 = Dense(128 * 7 * 7, 256)
        self.Fc2 = Dense(256, 128)
        self.Fc3 = Dense(128, 6 * self.k)  

    def encode_state(self, x):
                                               # x: 3 x 224 ** 2
        self.z1 = self.Pool1(self.Conv1(x))       # 8 x 112 ** 2
        self.z2 = self.Pool2(self.Conv2(self.z1)) # 16 x 56 ** 2
        self.z3 = self.Pool3(self.Conv3(self.z2)) # 32 x 28 ** 2
        self.z4 = self.Pool4(self.Conv4(self.z3)) # 64 x 14 ** 2
        self.z5 = self.Pool5(self.Conv5(self.z4)) # 128 x 7 ** 2
        z = self.z5
        return z

    def decode_mask(self, z):
                                               # z: 128 x 7 ** 2 
        self.m1 = self.Deconv1(z)                 # 64 x 14 ** 2
        self.m2 = self.Deconv2(self.m1 + self.z4) # 32 x 28 ** 2
        self.m3 = self.Deconv3(self.m2 + self.z3) # 16 x 56 ** 2
        self.m4 = self.Deconv4(self.m3 + self.z2) # 8 x 112 ** 2
        self.m5 = self.Deconv5(self.m4 + self.z1) # k x 224 ** 2
        k_sum = torch.sum(self.m5, dim=1)
        m = self.m5 / k_sum 
        return m

    def decode_pose(self, z):
        self.s1 = self.Fc1(z.view(-1, 128 * 7 * 7))
        self.s2 = self.Fc2(self.s1)
        self.s3 = self.Fc3(self.s2)
        p = self.s3
        return p

    def forward(self, x):
        enc_state = self.encode_state(x)
        mask = self.decode_mask(enc_state)
        poses = self.decode_pose(enc_state)
        return mask, poses


class PoseTransitionNetwork(nn.Module):
    def __init__(self, k, action_dim):
        super(PoseTransitionNetwork, self).__init__()
        self.k = k
        self.action_dim = action_dim

        # Pose change predictor
        self.Fc1_a = Dense(6 * self.k, 128)
        self.Fc2_a = Dense(128, 256)
        self.Fc1_b = Dense(self.action_dim, 128)
        self.Fc2_b = Dense(128, 256)
        self.Fc1_ab = Dense(512, 128)
        self.Fc2_ab = Dense(128, 64)
        self.Fc3_ab = Dense(64, 6 * self.k)

    def predict_pose_change(self, p, u):
        p_enc = self.Fc2_a(self.Fc1_a(p))
        u_enc = self.Fc2_b(self.Fc1_b(u))
        l = torch.cat((p_enc, u_enc), dim=-1)
        delta_p = self.Fc3_ab(self.Fc2_ab(self.Fc1_ab(l)))
        return delta_p
    
    def forward(self, poses, action):
        delta_poses = self.predict_pose_change(poses, action)
        return delta_poses

class TransformNetwork(nn.Module):
    def __init__(self, k, action_dim, gamma=1.0, sigma=0.5, training=True):
        super(TransformNetwork, self).__init__()
        self.k = k
        self.action_dim = action_dim
        self.pose_transition_network = PoseTransitionNetwork(k, action_dim)
        self.gamma = gamma
        self.sigma = sigma
        self.training = training

    def add_delta_poses(self, poses, delta_poses):
        poses_unflattened = poses.view(self.k, 6)
        delta_poses_unflattened = poses.view(self.k, 6)
        T = poses_unflattened[:, :3]
        axang = poses_unflattened[:, 3:]
        v = axang[:, :3] 
        R = axisAngleToRotationMatrix_batched(v)

        delta_T = delta_poses_unflattened[:, :3]
        delta_axang = delta_poses_unflattened[:, 3:]
        delta_v = axang[:, :3] 
        delta_R = axisAngleToRotationMatrix_batched(v)        

        axang_new = rotationMatrixToAxisAngle_batched(torch.matmul(delta_R, R)) # check if correct if R is batched
        pose_new = torch.stack((T + delta_T, axang_new), dim=1)
        pose_new = pose_new.view(-1).unsqueeze(0)
        return pose_new # (1, K * 6)

    def sharpen_mask_weights(self, mask):

        W = mask.shape[-1] 
        H = mask.shape[-2]
        mask = mask.view(-1, self.k, H * W)
        eps = torch.normal(mean=torch.zeros(self.k, H * W,device=mask.device), std=torch.ones(self.k, H * W,device=mask.device) * self.sigma ** 2)
        mask = (mask + eps.unsqueeze(0)) ** self.gamma
        k_sum = torch.sum(mask, dim=1)
        mask = mask / k_sum # sum of all mask weights per pixel equals 1    
        mask = mask.view(-1, self.k, H, W)
        if not self.training:
            ind = torch.max(mask, 1)[1]
            masmaskk_s = torch.zeros(mask.shape)
            mask[torch.arange(len(ind)), ind] = 1.0

        return mask

    def transform_point_cloud(self, x, mask, delta_poses):
        H = x.shape[-2]
        W = x.shape[-1]

        delta_poses_unflattened = delta_poses.view(self.k, 6)
        x = x.view(3, -1) # flatten image (H*W)
        # get axang of current poses (poses is of dimension  6 * k, where k is the number of obj in scene)
        delta_T = delta_poses_unflattened[:, :3]
        delta_axang = delta_poses_unflattened[:, 3:]
        delta_v = delta_axang[:, :3] 
        delta_R = axisAngleToRotationMatrix_batched(delta_v)        
        
        m = self.sharpen_mask_weights(mask)
        m = mask.view(self.k, -1)
        m = m.unsqueeze(-1).expand(self.k, m.shape[1], 3).transpose(2, 1)
        soft_transformed = torch.mul(m, torch.matmul(delta_R, x) + delta_T.unsqueeze(-1).expand(self.k, 3, x.shape[-1]))

        x_new = torch.sum(soft_transformed, dim=0)
        x_new = x_new.view(3, H, W)
        return x_new

    def forward(self, x, mask, poses, action):
        delta_poses = self.pose_transition_network(poses, action)
        poses_new = self.add_delta_poses(poses, delta_poses)
        x_new = self.transform_point_cloud(x, mask, delta_poses)
        return x_new, poses_new 

class ProjectionLayer(nn.Module):
    def __init__(self, fx, fy, cx, cy):
        super(ProjectionLayer, self).__init__()
        self.fx = 917.6631469726562
        self.fy = 917.4476318359375
        self.cx = 956.4253540039062
        self.cy = 553.5712280273438

    def forward(self, P_t, P_t1):
        X_t, Y_t, Z_t = P_t[:, 0, :, :], P_t[:, 1, :, :], P_t[:, 2, :, :]
        X_t1, Y_t1, Z_t1 = P_t1[:, 0, :, :], P_t1[:, 1, :, :], P_t1[:, 2, :, :]

        u_t = (X_t * self.fx) / Z_t + self.cx
        v_t = (Y_t * self.fy) / Z_t + self.cy

        u_t1 = (X_t1 * self.fx) / Z_t1 + self.cx
        v_t1 = (Y_t1 * self.fy) / Z_t1 + self.cy

        flow_u = u_t1 - u_t
        flow_v = v_t1 - v_t

        w_t = torch.stack((flow_u, flow_v), dim=1)  # Shape: [batch_size, 2, H, W]

        return w_t


class ForwardModel(nn.Module):
    def __init__(self, k, action_dim, gamma=1.0, sigma=0.5, training=True):
        super(ForwardModel, self).__init__()
        self.k = k
        self.action_dim = action_dim
        self.gamma = gamma
        self.sigma = sigma
        self.training = training

        self.transformer = TransformNetwork(k, action_dim)
        self.encoder = PoseAndMaskEncoder(k)

        fx = 917.6631469726562
        fy = 917.4476318359375
        cx = 956.4253540039062
        cy = 553.5712280273438
        self.projection_layer = ProjectionLayer(fx, fy, cx, cy)

    def forward(self, x, action):
        mask, poses = self.encoder(x)
        x_new, poses_new = self.transformer(x, mask, poses, action)
        optical_flow = self.projection_layer(x,x_new.unsqueeze(0))
        return poses_new.unsqueeze(0), x_new.unsqueeze(0) ,optical_flow.unsqueeze(0)


def handle_invalid_values(point_clouds):
    point_clouds = torch.nan_to_num(point_clouds, nan=0.0)
    point_clouds[point_clouds == float('inf')] = 1e6
    point_clouds[point_clouds == float('-inf')] = -1e6

    point_clouds = torch.clamp(point_clouds, min=-1e3, max=1e3) #iiiii


    return point_clouds

def normalize_point_clouds(point_clouds):

    point_clouds = handle_invalid_values(point_clouds)


    reshaped = point_clouds.view(point_clouds.size(0), 3, -1)  # Shape: [batch_size, 3, H*W]
    
    # Calculate mean and standard deviation
    mean = reshaped.mean(dim=2, keepdim=True)  # Shape: [batch_size, 3, 1]
    std = reshaped.std(dim=2, keepdim=True)    # Shape: [batch_size, 3, 1]

    # print("Mean:", mean)
    # print("Std:", std)
    
    
    normalized = (reshaped - mean) / (std + 1e-6)  
    
    return normalized  

def compute_chamfer_distance(preds, targets):
    if preds.shape[1] != 3 or targets.shape[1] != 3:
        raise ValueError("Input tensors must have shape [1, 3, H, W]")

    preds = normalize_point_clouds(preds)
    targets = normalize_point_clouds(targets)
    
    points1 = preds.view(1, 3, -1).transpose(1, 2)  # Shape: [1, H*W, 3]
    points2 = targets.view(1, 3, -1).transpose(1, 2)  # Shape: [1, H*W, 3]

    
    if torch.isnan(points1).any() or torch.isnan(points2).any():
        print("NaN detected in input points after normalization.")
        return None

    if torch.isinf(points1).any() or torch.isinf(points2).any():
        print("Infinite values detected in input points after normalization.")
        return None
    

    # Calculate the Chamfer distance
    dist1, dist2 = chamfer_distance(points1, points2)

    # Compute Chamfer distance loss
    if dist1 is not None and dist2 is not None:
        chamfer_loss = (dist1.mean()+1e-6 + dist2.mean()+1e-6) / 2
    else:
        chamfer_loss = torch.tensor(1e-6, device=preds.device)


    return chamfer_loss

def check_invalid_values(tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        return True
    return False

def handle_invalid_values(tensor):
    return torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)

class InverseDynamicsModel(nn.Module):
    def __init__(self, k, encoder):
        """
        Inverse dynamics model that predicts the poke action between two 3D point clouds.
        
        Args:
        - k: Number of objects in the scene.
        - encoder: An instance of PoseAndMaskEncoder to encode the point clouds.
        """
        super(InverseDynamicsModel, self).__init__()
        
        self.k = k
        self.encoder = encoder 
        
        
        self.heatmap_A = nn.Conv2d(128, 2, kernel_size=1, stride=1)  
        self.heatmap_B = nn.Conv2d(128, 2, kernel_size=1, stride=1)
        
        
        self.fc1 = nn.Linear(128 * 7 * 7 * 2, 256)  
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 4) 

    def forward(self, P_t, P_t1):
        encoded_Pt = self.encoder.encode_state(P_t)      
        encoded_Pt1 = self.encoder.encode_state(P_t1) 

        heatmap_A = self.heatmap_A(encoded_Pt)          
        heatmap_B = self.heatmap_B(encoded_Pt1)          
        

        flattened_Pt = encoded_Pt.view(encoded_Pt.size(0), -1)
        flattened_Pt1 = encoded_Pt1.view(encoded_Pt1.size(0), -1)
        combined_features = torch.cat((flattened_Pt, flattened_Pt1), dim=1)


        # Predict the poke action using fully connected layers
        action = F.relu(self.fc1(combined_features))
        action = F.relu(self.fc2(action))
        action = self.fc3(action)  
        
        return heatmap_A, heatmap_B, action

def prepare_heatmap(heatmap):
    heatmap_np = heatmap.squeeze().detach().cpu().numpy()
    heatmap_np = (heatmap_np - np.min(heatmap_np)) / (np.max(heatmap_np) - np.min(heatmap_np))
    return heatmap_np

def visualize_heatmaps(heatmap_A, heatmap_B):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
   
    heatmap_A_np = prepare_heatmap(heatmap_A[0, 0, :, :])
    heatmap_B_np = prepare_heatmap(heatmap_B[0, 0, :, :])
    
    
    sns.heatmap(heatmap_A_np, cmap='viridis', ax=axes[0], cbar=True)
    axes[0].set_title('Heatmap Â_t (Start Position)')
    
    sns.heatmap(heatmap_B_np, cmap='viridis', ax=axes[1], cbar=True)
    axes[1].set_title('Heatmap B̂_t (End Position)')
    
    plt.show()



def cross_entropy_between_heatmaps(heatmap_A, heatmap_B):
    prob_A = F.softmax(heatmap_A, dim=1)  
    prob_B = F.softmax(heatmap_B, dim=1) 
    loss = -torch.sum(prob_B * torch.log(prob_A + 1e-10))  

    return loss.mean()  


class Hind4sightModel(nn.Module):
    def __init__(self, k, action_dim):
        super(Hind4sightModel, self).__init__()
        self.k = k
        self.action_dim = action_dim

        # Initialize forward and inverse models
        self.forward_model = ForwardModel(k, action_dim)
        self.inverse_model = InverseDynamicsModel(k, encoder=PoseAndMaskEncoder(k))

    def forward(self, P_t, P_t1, action):
        # Forward Dynamics (Predicting next state from current state and action)
        poses_new, P_t1_predicted, optical_flow = self.forward_model(P_t, action)
        # Inverse Dynamics (Predicting action from two consecutive states)
        heatmap_A, heatmap_B, predicted_action = self.inverse_model(P_t.squeeze(1), P_t1.squeeze(1))

        return poses_new, P_t1_predicted, optical_flow, heatmap_A, heatmap_B, predicted_action


class PointCloudDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)
        self.keys = list(self.data.keys())
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        pointcloud1_tensor, pointcloud2_tensor, action_tensor, flow_tensor = self.data[key]
        return pointcloud1_tensor, pointcloud2_tensor, action_tensor, flow_tensor

dataset = PointCloudDataset('/home/oviya/coro_project/final/all_pointcloud_tensors_unbatched.pth')
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)

# Initialize the Hind4sight model
k = 5
action_dim = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Hind4sightModel(k, action_dim).to(device)

# Define optimizers and loss functions
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion_action = nn.MSELoss()
criterion_cross_entropy = nn.CrossEntropyLoss()

num_epochs = 50

for epoch in range(num_epochs):
    epoch_loss = 0.0
    model.train()

    for P_t, P_t1, action_tensor, flow_tensor in data_loader:
        P_t = P_t.to(device).float()
        P_t1 = P_t1.to(device).float()
        action_tensor = action_tensor.to(device).float()
        P_t = P_t.squeeze(0)
        action = action_tensor.float()
        # Zero gradients
        optimizer.zero_grad()
        # Forward pass
        poses_new, P_t1_predicted, optical_flow, heatmap_A, heatmap_B, predicted_action = model(P_t, P_t1, action_tensor)
        poses_new.to(device)      
        P_t1_predicted.to(device)

        target_poses = P_t1.float().squeeze(0)

        
        chamfer_loss = compute_chamfer_distance(P_t1_predicted, target_poses)

        P_t1_predicted = handle_invalid_values(P_t1_predicted)
        target_poses = handle_invalid_values(target_poses)

        if check_invalid_values(P_t1_predicted) or check_invalid_values(target_poses):
            print("Invalid values found in the inputs for MSE loss. Skipping this batch.")
            continue

        mse_loss = criterion_action(P_t1_predicted, target_poses)
        mse_loss = torch.clamp(mse_loss, min=0, max=1e3)
        
        entropy_loss=cross_entropy_between_heatmaps(heatmap_A, heatmap_B)
        total_loss = chamfer_loss + mse_loss+entropy_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += total_loss.item()
    
    avg_epoch_loss = epoch_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataset)}")

torch.cuda.empty_cache()
# After training, save the model
torch.save(model.state_dict(), '/home/oviya/coro_project/final/hind4sight.pth')