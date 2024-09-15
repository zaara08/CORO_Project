import torch
from torch.autograd import Function
from torch.nn import Module
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F

class DepthImageToDense3DPointsFunction(Function):
    @staticmethod
    def forward(ctx, depth, height, width, base_grid, fy, fx, cy, cx):
        # Check dimensions (B x 1 x H x W)
        batch_size, num_channels, num_rows, num_cols = depth.size()
        assert(num_channels == 1)
        assert(num_rows == height)
        assert(num_cols == width)

        
        ctx.save_for_backward(depth)
        ctx.base_grid = base_grid
        ctx.height = height
        ctx.width = width

        
        output = depth.new_zeros((batch_size, 3, num_rows, num_cols))
        xy, z = output.narrow(1, 0, 2), output.narrow(1, 2, 1)
        z.copy_(depth)  
        xy.copy_(depth.expand_as(xy) * ctx.base_grid.narrow(1, 0, 2).expand_as(xy))  # [x*z, y*z, z]

        return output

    @staticmethod
    def backward(ctx, grad_output):
        depth, = ctx.saved_tensors
        base_grid = ctx.base_grid

        # Compute grad input
        grad_input = (base_grid.expand_as(grad_output) * grad_output).sum(1)

        return grad_input, None, None, None, None, None, None  

class DepthImageToDense3DPoints(Module):
    def __init__(self, height, width,
                 fy=589.3664541825391 * 0.5,
                 fx=589.3664541825391 * 0.5,
                 cy=240.5 * 0.5,
                 cx=320.5 * 0.5):
        super(DepthImageToDense3DPoints, self).__init__()
        self.height = height
        self.width = width
        self.base_grid = None
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def forward(self, input):
        if self.base_grid is None:
            self.base_grid = torch.ones(1, 3, self.height, self.width).type_as(input.data)  # (x,y,1)
            for j in range(self.width): 
                self.base_grid[:, 0, :, j].fill_((j - self.cx) / self.fx)
            for i in range(self.height): 
                self.base_grid[:, 1, i, :].fill_((i - self.cy) / self.fy)

        
        return DepthImageToDense3DPointsFunction.apply(input, self.height, self.width, self.base_grid, self.fy, self.fx, self.cy, self.cx)






def load_depth_image(depth_path):
    """Load depth image as a numpy array."""
    depth_image = np.array(Image.open(depth_path)) / 1000.0  
    return depth_image

def load_action_data(action_path):
    """Load action data from action.txt."""
    with open(action_path, 'r') as file:
        actions = file.readlines()
    
    action_tensor = torch.tensor(
        [float(value) for action in actions for value in action.strip().split()],
        dtype=torch.float32
    )
    
    return action_tensor

def load_flow_image(flow_path):
    """Load the RGB image representing optical flow."""
    flow_image = np.array(Image.open(flow_path))  
    return flow_image

def rgb_to_flow_tensor(flow_image):
    """Convert RGB flow image to optical flow tensor."""
    # Normalize the image to [0, 1]
    flow_image = flow_image / 255.0

    # Convert to tensor and reorder to [3, H, W]
    flow_tensor = torch.tensor(flow_image, dtype=torch.float32).permute(2, 0, 1)

    # Split channels
    u = flow_tensor[0, :, :]  # Horizontal flow
    v = flow_tensor[1, :, :]  # Vertical flow

    # Convert to range [-1, 1]
    u = u * 2 - 1
    v = v * 2 - 1

    return u, v

def create_flow_tensor(u, v):
    """Combine horizontal and vertical flow components into a single tensor."""
    flow = torch.stack([u, v], dim=0)  
    return flow


def load_flow_image(flow_path):
    """Load the RGB image representing optical flow and convert it to a tensor."""
    flow_image = np.array(Image.open(flow_path).convert('RGB'), dtype=np.float32)  # Convert to float32

    flow_image /= 255.0

    flow_tensor = torch.tensor(flow_image, dtype=torch.float32).permute(2, 0, 1)  # Shape: [C, H, W]


    u = flow_tensor[0, :, :] * 2 - 1  
    v = flow_tensor[1, :, :] * 2 - 1 


    flow = torch.stack([u, v], dim=0)


    flow = flow.unsqueeze(0)  
    resized_tensor = F.interpolate(flow, size=(224, 224), mode='bilinear', align_corners=True)

    return resized_tensor


def load_rgb_image(rgb_path):
    """Load RGB image and convert to tensor."""
    rgb_image = Image.open(rgb_path).convert('RGB')  # Open and convert to RGB
    transform = transforms.ToTensor()
    rgb_tensor = transform(rgb_image)
    # print("before",rgb_tensor.shape)
    rgb_tensor=rgb_tensor.unsqueeze(0)
    # print("before111",rgb_tensor.shape)
    rgb_tensor = F.interpolate(rgb_tensor, size=(224, 224), mode='bilinear', align_corners=True)
    # print("rgb_tensor",rgb_tensor.shape)
    return rgb_tensor

def process_episode(episode_folder):
    depth_1_path = os.path.join(episode_folder, 'depth_1.png')
    depth_2_path = os.path.join(episode_folder, 'depth_2.png')
    action_path = os.path.join(episode_folder, 'action')
    flow_path = os.path.join(episode_folder, 'flow.png')

    
    depth_1 = load_depth_image(depth_1_path)
    depth_2 = load_depth_image(depth_2_path)

    
    action_tensor = load_action_data(action_path)

    
    flow_tensor = load_flow_image(flow_path)


    depth_1_tensor = torch.tensor(depth_1).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    depth_2_tensor = torch.tensor(depth_2).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    
    height, width = depth_1_tensor.shape[2], depth_1_tensor.shape[3]
    depth_to_pointcloud = DepthImageToDense3DPoints(height=height, width=width)

    
    pointcloud1_tensor = depth_to_pointcloud(depth_1_tensor)
    pointcloud2_tensor = depth_to_pointcloud(depth_2_tensor)
    
    
    pointcloud1_tensor= F.interpolate(pointcloud1_tensor, size=(224, 224), mode='bilinear', align_corners=True)
    pointcloud2_tensor= F.interpolate(pointcloud2_tensor, size=(224, 224), mode='bilinear', align_corners=True)
    print("pointcloud1_tensor",pointcloud1_tensor.shape)
    print("pointcloud2_tensor",pointcloud2_tensor.shape)
    return pointcloud1_tensor, pointcloud2_tensor, action_tensor, flow_tensor #, rgb_1_tensor, rgb_2_tensor

def process_all_episodes(base_folder):
    """Process all episodes in the base folder."""
    episode_folders = [os.path.join(base_folder, d) for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]

    all_pointclouds = {}
    for episode_folder in episode_folders:
        episode_name = os.path.basename(episode_folder)
        print(f"Processing {episode_name}...")
        pointcloud1_tensor, pointcloud2_tensor, action_tensor, flow_tensor  = process_episode(episode_folder) #rgb_1_tensor, rgb_2_tensor
        all_pointclouds[episode_name] = (pointcloud1_tensor, pointcloud2_tensor, action_tensor, flow_tensor) #rgb_1_tensor, rgb_2_tensor

    return all_pointclouds


base_folder = '/home/oviya/coro_project/trial_real'


all_pointcloud_tensors = process_all_episodes(base_folder)
save_path = '/home/oviya/coro_project/final/all_pointcloud_tensors_unbatched.pth'
torch.save(all_pointcloud_tensors, save_path)