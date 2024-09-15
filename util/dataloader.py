import torchvision.transforms as transforms
from PIL import Image
import os
import torch
import numpy as np

def load_image(image_path, grayscale=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(), 
    ])
    if grayscale:

        image = Image.open(image_path).convert('L') 
    else: 
        image = Image.open(image_path).convert('RGB')
    return transform(image).permute(0,1,2).numpy()  


class EpisodeDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, device='cpu'):
        self.root_dir = root_dir
        self.episodes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.device = device
        self.camera_params_dir = os.path.join(root_dir, 'intrinsics.txt')
        
    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        episode_dir = os.path.join(self.root_dir, self.episodes[idx])

        action = torch.tensor(np.loadtxt(os.path.join(episode_dir, 'action'))).to(self.device)
        action_ang = torch.tensor(np.loadtxt(os.path.join(episode_dir, 'action_ang'))).to(self.device)
        crop_info = torch.tensor(np.loadtxt(os.path.join(episode_dir, 'crop_info'))).to(self.device)
        depth_1 = load_image(os.path.join(episode_dir, 'depth_1.png'), grayscale=True)
        depth_1 = torch.tensor(depth_1, device=self.device).permute(0,1,2)
        depth_2 = load_image(os.path.join(episode_dir, 'depth_2.png'), grayscale=True)
        depth_2 = torch.tensor(depth_2, device=self.device).permute(0,1,2)
        try:
            flow = load_image(os.path.join(episode_dir, 'flow.png'))
            flow = torch.tensor(flow, device=self.device).permute(0,1,2)
        except:
            flow = np.zeros((224,224,3))
            flow = torch.tensor(flow, device=self.device).permute(0,1,2)
        rgb_1 = load_image(os.path.join(episode_dir, 'rgb_1.png'))
        rgb_1 = torch.tensor(rgb_1, device=self.device).permute(0,1,2)

        rgb_2 = load_image(os.path.join(episode_dir, 'rgb_2.png'))
        rgb_2 = torch.tensor(rgb_2, device=self.device).permute(0,1,2)
        return {
            'action': action,
            'action_ang': action_ang,
            'crop_info': crop_info,
            'depth_1': depth_1,
            'depth_2': depth_2,
            'flow': flow,
            'rgb_1': rgb_1,
            'rgb_2': rgb_2,
            'pt1': self.get_point_clouds(idx)[0],
            'pt2': self.get_point_clouds(idx)[1],
            'rgb1': self.get_point_clouds(idx)[2],
            'rgb2': self.get_point_clouds(idx)[3]
        }
    
    def get_images_numpy(self, idx):
        episode_dir = os.path.join(self.root_dir, self.episodes[idx])
        depth_1 = load_image(os.path.join(episode_dir, 'depth_1.png'))
        depth_2 = load_image(os.path.join(episode_dir, 'depth_2.png'))
        flow = load_image(os.path.join(episode_dir, 'flow.png'))
        rgb_1 = load_image(os.path.join(episode_dir, 'rgb_1.png'))
        rgb_2 = load_image(os.path.join(episode_dir, 'rgb_2.png'))

        return depth_1, depth_2, flow, rgb_1, rgb_2
    
    def get_images_tensor(self, idx):
        episode_dir = os.path.join(self.root_dir, self.episodes[idx])
        depth_1 = load_image(os.path.join(episode_dir, 'depth_1.png'))
        depth_2 = load_image(os.path.join(episode_dir, 'depth_2.png'))
        flow = load_image(os.path.join(episode_dir, 'flow.png'))
        rgb_1 = load_image(os.path.join(episode_dir, 'rgb_1.png'))
        rgb_2 = load_image(os.path.join(episode_dir, 'rgb_2.png'))

        depth_1 = torch.tensor(depth_1, device=self.device).permute(0,1,2)
        depth_2 = torch.tensor(depth_2, device=self.device).permute(0,1,2)
        flow = torch.tensor(flow, device=self.device).permute(0,1,2)
        rgb_1 = torch.tensor(rgb_1, device=self.device).permute(0,1,2)
        rgb_2 = torch.tensor(rgb_2, device=self.device).permute(0,1,2)

        return torch.tensor(depth_1).permute(0,1,2), torch.tensor(depth_2).permute(0,1,2), \
               torch.tensor(flow).permute(0,1,2), torch.tensor(rgb_1).permute(0,1,2), \
               torch.tensor(rgb_2).permute(0,1,2)
    
    
    def get_actions_numpy(self, idx):
        episode_dir = os.path.join(self.root_dir, self.episodes[idx])
        action = np.loadtxt(os.path.join(episode_dir, 'action'))
        action_ang = np.loadtxt(os.path.join(episode_dir, 'action_ang'))
        crop_info = np.loadtxt(os.path.join(episode_dir, 'crop_info'))

        return action, action_ang, crop_info
    
    def get_actions_tensor(self, idx):
        episode_dir = os.path.join(self.root_dir, self.episodes[idx])
        action = torch.tensor(np.loadtxt(os.path.join(episode_dir, 'action')), dtype=torch.float32)
        action_ang = torch.tensor(np.loadtxt(os.path.join(episode_dir, 'action_ang')) , dtype=torch.float32)
        crop_info = torch.tensor(np.loadtxt(os.path.join(episode_dir, 'crop_info')) , dtype=torch.float32)
        action = torch.tensor(action, device=self.device)
        action_ang = torch.tensor(action_ang, device=self.device)
        crop_info = torch.tensor(crop_info, device=self.device)

        return action, action_ang, crop_info

    def get_intrinsics(self):
       
        with open(self.camera_params_dir, 'r') as f:
            lines = f.readlines()
            intrinsics = lines[1].split(' ')
        """ 
        fx fy cx cy offx offy sx sy
        917.6631469726562 917.4476318359375 956.4253540039062 553.5712280273438 575 285 1.9375 1.9416666666666667
        """
        return {
            'fx': float(intrinsics[0]),
            'fy': float(intrinsics[1]),
            'cx': float(intrinsics[2]),
            'cy': float(intrinsics[3]),
            'offx': float(intrinsics[4]),
            'offy': float(intrinsics[5]),
            'sx': float(intrinsics[6]),
            'sy': float(intrinsics[7])
        }

    def get_point_clouds(self, idx):
        episode_dir = os.path.join(self.root_dir, self.episodes[idx])
        depth1, depth2, flow, rgb1, rgb2 = self.get_images_tensor(idx)
        intrinsics = self.get_intrinsics()

        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['cx']
        cy = intrinsics['cy']
        offx = intrinsics['offx']
        offy = intrinsics['offy']
        sx = intrinsics['sx']
        sy = intrinsics['sy']
     
        depth1 = depth1.squeeze(0).squeeze(0).cpu().numpy()[0]
        depth2 = depth2.squeeze(0).squeeze(0).cpu().numpy()[0]

        rgb1 = rgb1.squeeze(0).squeeze(0).cpu().numpy()
        rgb2 = rgb2.squeeze(0).squeeze(0).cpu().numpy()

        print("Depth1 shape: ", depth1.shape)

        h, w = depth1.shape

        x = np.arange(0, w)
        y = np.arange(0, h)

        xx, yy = np.meshgrid(x, y)

        x1 = (xx - cx) * depth1 / fx
        y1 = (yy - cy) * depth1 / fy
        z1 = depth1
        x2 = (xx - cx) * depth2 / fx
        y2 = (yy - cy) * depth2 / fy
        z2 = depth2

        pt1 = np.stack([x1, y1, z1], axis=-1) 
        pt2 = np.stack([x2, y2, z2], axis=-1) 
        pt1 = torch.tensor(pt1, device=self.device).permute(2,0,1)
        pt2 = torch.tensor(pt2, device=self.device).permute(2,0,1)
        print("Point clouds shape: ", pt1.shape, pt2.shape)
        return pt1, pt2, rgb1, rgb2
