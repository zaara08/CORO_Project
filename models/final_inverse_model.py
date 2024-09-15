import torch
import torch.nn as nn
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

from copy import deepcopy as copy
import math
import sys
from pdb import set_trace as st 


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



encoder = PoseAndMaskEncoder(k=5)
model = InverseDynamicsModel(k=5, encoder=encoder)


criterion_action = nn.MSELoss()


criterion_cross_entropy = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.00001)


all_pointcloud_tensors=torch.load('/home/oviya/coro_project/final/all_pointcloud_tensors_unbatched.pth') #'/home/oviya/coro_project/all_pointcloud_tensors_batched_trial.pth'
num_epochs=50 
for epoch in range(num_epochs):
    total_loss = 0.0
    model.train()  

    for episode_name, (pointcloud1_tensor, pointcloud2_tensor, action_tensor, flow_tensor) in all_pointcloud_tensors.items(): 
       
        P_t = pointcloud1_tensor.float().squeeze(1) 
        P_t1 = pointcloud1_tensor.float().squeeze(1)  
        
        
        optimizer.zero_grad() 

        heatmap_A, heatmap_B, action = model(P_t, P_t1) 
        
        
        total_loss = cross_entropy_between_heatmaps(heatmap_A, heatmap_B)


        total_loss.backward()
        optimizer.step()


    total_loss += total_loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(all_pointcloud_tensors)}")

torch.save(model.state_dict(), '/home/oviya/coro_project/final/inverse_dynamics_model_unsupervised_trial.pth')
