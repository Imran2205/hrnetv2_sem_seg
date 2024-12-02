import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from imageio import imread
from PIL import Image
import cv2
import os
from scipy.io import loadmat
from tqdm import tqdm
from torch.nn import functional as F
import os

import cv2
import numpy as np
import random

import torch
from torch.nn import functional as F
from torch.utils import data

from config import config_hrnet_v2 as config


class UWSDataLoader(torch.utils.data.Dataset):
    def __init__(self, 
                 output_image_height=700, 
                 images=None,
                 masks=None, 
                 transform=None,
                 channel_values=None):
        """
        output_image_height: output image height in pixels
        images: list of images
        masks: list of masks
        transform: instance of transform.Compose initialized with a list of transformations
        channel_values: A dictionary containing the train_id and color of each class in the segmentation mask
        """
        self.output_image_height = output_image_height
        self.images = images
        self.masks = masks
        self.transform = transform
        self.channel_values = channel_values

        if not self.channel_values:
            """
                Train ID 255 is used for representing unlabeled class, which is ignored during training.
                We follow suggestions from the cityscape dataset train IDs
                Please refer to cityscape dataset train IDs for details: 
                https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
            """
            self.label_dictionary = {
                0:  {'name': 'unlabeled',   'train_id': 255, 'color': (0,   0,   0)},            
                1:  {'name': 'crab',        'train_id': 0,   'color': (255, 178, 204)},
                2:  {'name': 'crocodile',   'train_id': 1,   'color': (0, 0, 128)},
                3:  {'name': 'dolphin',     'train_id': 2,   'color': (0, 0, 178)},
                4:  {'name': 'frog',        'train_id': 3,   'color': (51, 51, 51)},
                5:  {'name': 'nettles',     'train_id': 4,   'color': (0, 0, 0)},
                6:  {'name': 'octopus',     'train_id': 5,   'color': (51, 306, 51)},
                7:  {'name': 'otter',       'train_id': 6,   'color': (102, 102, 102)},
                8:  {'name': 'penguin',     'train_id': 7,   'color': (10, 0, 255)},
                9:  {'name': 'polar_bear',  'train_id': 8,   'color': (255, 178, 102)},
                10: {'name': 'sea_anemone', 'train_id': 9,  'color':  (153, 255, 255)},
                11: {'name': 'sea_urchin',  'train_id': 10,  'color': (0, 255, 255)},
                12: {'name': 'seahorse',    'train_id': 11,  'color': (255, 153, 153)},
                13: {'name': 'seal',        'train_id': 12,  'color': (255, 0, 0)},
                14: {'name': 'shark',       'train_id': 13,  'color': (178, 178, 0)},
                15: {'name': 'shrimp',      'train_id': 14,  'color': (255, 102, 178)},
                16: {'name': 'star_fish',   'train_id': 15,  'color': (153, 204, 255)},
                17: {'name': 'stingray',    'train_id': 16,  'color': (255, 153, 178)},
                18: {'name': 'squid',       'train_id': 17,  'color': (229, 0, 0)},
                19: {'name': 'turtle',      'train_id': 18,  'color': (0, 153, 0)},
                20: {'name': 'whale',       'train_id': 19,  'color': (0, 229, 77)},
                21: {'name': 'nudibranch',  'train_id': 20,  'color': (242, 243, 245)},
                22: {'name': 'coral',       'train_id': 21,  'color': (0, 0, 77)},
                23: {'name': 'rock',        'train_id': 22,  'color': (0, 178, 0)},
                24: {'name': 'water',       'train_id': 23,  'color': (255, 77, 77)},
                25: {'name': 'sand',        'train_id': 24,  'color': (178, 0, 0)},
                26: {'name': 'plant',       'train_id': 25,  'color': (255, 178, 255)},
                27: {'name': 'human',       'train_id': 26,  'color': (128, 128, 0)},
                28: {'name': 'reef',        'train_id': 27,  'color': (0, 0, 255)},
                29: {'name': 'others',      'train_id': 28,  'color': (178, 178, 178)},
                30: {'name': 'dynamic',     'train_id': 29,  'color': (0, 77, 0)}, ## begining of UWSv2 new semantic categories
                31: {'name': 'beaver',      'train_id': 30,  'color': (151, 191, 201)},
                32: {'name': 'duck',        'train_id': 31,  'color': (153, 102, 51)},
                33: {'name': 'dugong',      'train_id': 32,  'color': (229, 0, 229)},
                34: {'name': 'hippo',       'train_id': 33,  'color': (255, 255, 178)},
                35: {'name': 'lobster',     'train_id': 34,  'color': (222, 128, 4)},
                36: {'name': 'platypus',    'train_id': 35,  'color': (102, 87, 110)},
                37: {'name': 'nautilus',    'train_id': 36,  'color': (229, 229, 0)},
                38: {'name': 'sea_cucumber','train_id': 37,  'color': (229, 255, 255)},
                39: {'name': 'sea_lion',    'train_id': 38,  'color': (173, 173, 0)},
                40: {'name': 'sea_snake',   'train_id': 39,  'color': (0, 0, 102)},
                41: {'name': 'barracouta',  'train_id': 40,  'color': (77, 0, 0)},
                42: {'name': 'billfish',    'train_id': 41,  'color': (170, 184, 90)},
                43: {'name': 'coho',        'train_id': 42,  'color': (174, 230, 187)},
                44: {'name': 'eel',         'train_id': 43,  'color': (0, 178, 178)},
                45: {'name': 'goldfish',    'train_id': 44,  'color': (173, 121, 0)},
                46: {'name': 'jellyfish',   'train_id': 45,  'color': (97, 194, 157)},
                47: {'name': 'lionfish',    'train_id': 46,  'color': (0, 128, 255)},
                48: {'name': 'puffer',      'train_id': 47,  'color': (87, 106, 110)},
                49: {'name': 'rock_beauty', 'train_id': 48,  'color': (142, 173, 0)},
                50: {'name': 'sturgeon',    'train_id': 49,  'color': (27, 71, 74)},
                51: {'name': 'tench',       'train_id': 50,  'color': (209, 88, 88)}
            }
        else:
            self.label_dictionary = self.channel_values

        self.length = len(self.images)
        
        print(f"UWSDataloader with {len(self.label_dictionary)} objects")
        print(f"UWSDataloader label and color mapping: {self.label_dictionary}")

        if self.length == 0:
            raise FileNotFoundError('No dataset files found')

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_label_dict(self):
        return self.label_dictionary

    def get_image_nd_label(self, index):
        image = self.images[index]
        label = self.masks[index]
        # if 51 in list(np.unique(label)):
        #     Image.fromarray(image.astype(np.uint8)).save("./error_rgb.png")
        #     Image.fromarray(label.astype(np.uint8)).save("./error_lbl.png")
        return image, label

    def __getitem__(self, index):
        img, label_image_gray = self.get_image_nd_label(index)

        if self.transform:
            image, label_image_gray = self.transform(img, label_image_gray)
        else:
            raise NotImplementedError("Transform not implemented...")

        return image, label_image_gray

    def __len__(self):
        return len(self.images)
