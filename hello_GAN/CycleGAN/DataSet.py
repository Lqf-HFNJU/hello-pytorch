from torch.utils.data.dataset import Dataset
import torchvision.transforms as transform
from PIL import Image
import glob
import os
import numpy as np
import random
import torch


class CreateDatasets(Dataset):
    def __init__(self, root_path, img_size, mode):
        if mode == 'train':
            A_img_path = os.path.join(root_path, 'trainA')
            B_img_path = os.path.join(root_path, 'trainB')
        elif mode == 'test':
            A_img_path = os.path.join(root_path, 'testA')
            B_img_path = os.path.join(root_path, 'testB')
        else:
            raise NotImplementedError('mode {} is error}'.format(mode))

        self.A_img_list = glob.glob(A_img_path + '/*.jpg')
        self.B_img_list = glob.glob(B_img_path + '/*.jpg')
        self.transform = transform.Compose([
            transform.ToTensor(),
            transform.Resize((img_size, img_size)),
            transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.A_img_list)

    def __getitem__(self, item):
        A_index = item % len(self.A_img_list)
        A_img = Image.open(self.A_img_list[A_index])
        B_index = np.random.randint(0, len(self.B_img_list) - 1)
        B_img = Image.open(self.B_img_list[B_index])
        A_img = self.transform(A_img)
        B_img = self.transform(B_img)
        return A_img, B_img


class ImagePool:
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Return images from the buffer.

        By 50% chance, the buffer will return input images.
        By 50% chance, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:  # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:  # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)  # collect all the images and return
        return return_images
