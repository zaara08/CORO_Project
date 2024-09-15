import torch
from torch.autograd import Function
from torch.nn import Module
import os
import numpy as np
import torch
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image





class DepthImageToDense3DPointsFunction(Function):
    @staticmethod
    def forward(ctx, depth, height, width, base_grid, fy, fx, cy, cx):
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
        z.copy_(depth)  # z = depth
        xy.copy_(depth.expand_as(xy) * ctx.base_grid.narrow(1, 0, 2).expand_as(xy))  # [x*z, y*z, z]

        return output

    @staticmethod
    def backward(ctx, grad_output):
        depth, = ctx.saved_tensors
        base_grid = ctx.base_grid

        grad_input = (base_grid.expand_as(grad_output) * grad_output).sum(1)

        return grad_input, None, None, None, None, None, None  # Return None for unused gradients

class DepthImageToDense3DPoints(Module):
    def __init__(self, height, width,
				 fy = 589.3664541825391 * 0.5,
				 fx = 589.3664541825391 * 0.5,
				 cy = 240.5 * 0.5,
				 cx = 320.5 * 0.5):
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
            for j in range(self.width):  # +x is increasing columns
                self.base_grid[:, 0, :, j].fill_((j - self.cx) / self.fx)
            for i in range(self.height):  # +y is increasing rows
                self.base_grid[:, 1, i, :].fill_((i - self.cy) / self.fy)


        return DepthImageToDense3DPointsFunction.apply(input, self.height, self.width, self.base_grid, self.fy, self.fx, self.cy, self.cx)



def load_depth_image(depth_path):
    depth_image = np.array(Image.open(depth_path)) / 1000.0  
    return depth_image


def load_action_data(action_path):
    with open(action_path, 'r') as file:
        actions = file.readlines()
    

    action_tensor = torch.tensor(
        [float(value) for action in actions for value in action.strip().split()],
        dtype=torch.float32
    )
    
    return action_tensor


def load_flow_image(flow_path):
    flow_image = np.array(Image.open(flow_path))  
    flow_tensor = torch.tensor(flow_image, dtype=torch.float32)  
    return flow_tensor


def process_episode(episode_folder):
    depth_1_path = os.path.join(episode_folder, 'depth_1.png')
    depth_2_path = os.path.join(episode_folder, 'depth_2.png')
    action_path = os.path.join(episode_folder, 'action')
    flow_path = os.path.join(episode_folder, 'flow.png')


    depth_1 = load_depth_image(depth_1_path)
    depth_2 = load_depth_image(depth_2_path)


    action_tensor = load_action_data(action_path)


    flow_tensor = load_flow_image(flow_path)


    depth_1_tensor = torch.tensor(depth_1).unsqueeze(0).unsqueeze(0)  
    depth_2_tensor = torch.tensor(depth_2).unsqueeze(0).unsqueeze(0)  

    height, width = depth_1_tensor.shape[2], depth_1_tensor.shape[3]
    depth_to_pointcloud = DepthImageToDense3DPoints(height=height, width=width)


    pointcloud1_tensor = depth_to_pointcloud(depth_1_tensor)
    pointcloud2_tensor = depth_to_pointcloud(depth_2_tensor)

    return pointcloud1_tensor, pointcloud2_tensor, action_tensor, flow_tensor



def process_all_episodes(base_folder):
   
    episode_folders = [os.path.join(base_folder, d) for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]

    all_pointclouds = {}
    for episode_folder in episode_folders:
        episode_name = os.path.basename(episode_folder)
        print(f"Processing {episode_name}...")
        pointcloud1_tensor, pointcloud2_tensor, action_tensor, flow_tensor = process_episode(episode_folder)
        all_pointclouds[episode_name] = (pointcloud1_tensor, pointcloud2_tensor, action_tensor, flow_tensor)

    return all_pointclouds


