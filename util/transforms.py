from skimage import io, transform
import numpy as np
import torch
from ipdb import set_trace as st 

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))

  
        points = list(np.where(landmarks == 1))
        points[0] = np.array(points[0] * new_h / h, dtype=np.int32)
        points[1] = np.array(points[1] * new_w / w, dtype=np.int32)
        landmarks = np.zeros((img.shape[0], img.shape[1]))

        landmarks[tuple(points)] = 1
        return {'image': img, 'label': landmarks}


class RandomCrop(object):


    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        points = list(np.where(landmarks == 1))
        points[0] = np.array(points[0] - top, dtype=np.int32)
        points[1] = np.array(points[1] - left, dtype=np.int32)
        landmarks = np.zeros((image.shape[0], image.shape[1]))
        landmarks[tuple(points)] = 1

        return {'image': image, 'label': landmarks}


class ToTensor(object):


    def __call__(self, sample):
        image, landmarks = sample['image'], sample['label']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(landmarks)}
