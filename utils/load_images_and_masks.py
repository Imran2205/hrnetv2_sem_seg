import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import glob


def load_images_and_masks(dataset_root, split, augment=False):
    """
    Load images and masks from a directory of images and masks.
    dataset_root: root directory of dataset
    split: train or val
    augment: whether to augment or not
    """
    images_files = glob.glob(
        os.path.join(
            dataset_root,
            split,
            'images',
            '*.png'
        )
    )
    masks_files = [
        os.path.join(dataset_root, split, 'labels', os.path.basename(m_i)) for m_i in images_files
    ]

    images = []
    masks = []

    if augment:
        for i_i_fl, img_fl in enumerate(tqdm(images_files)):
            images.append(np.array(
                Image.open(img_fl)
            ))
            masks.append(np.array(
                Image.open(masks_files[i_i_fl])
            ))

        dataset_len = len(images)
        np.random.seed(0)
        rand_n = list(np.random.randint(low=0, high=dataset_len, size=dataset_len // 2))
        for i in rand_n:
            im = images[i]
            target = masks[i]
            # perform horizontal flip
            images.append(np.fliplr(im))
            masks.append(np.fliplr(target))

        # shift right
        rand_n = list(np.random.randint(low=0, high=dataset_len, size=dataset_len // 4))
        for i in rand_n:
            shift = 20
            im = images[i]
            target = masks[i]

            im[:, shift:] = im[:, :-shift]
            target[:, shift:] = target[:, :-shift]
            images.append(im)
            masks.append(target)

        # shift left
        rand_n = list(np.random.randint(low=0, high=dataset_len, size=dataset_len // 4))
        for i in rand_n:
            shift = 20
            im = images[i]
            target = masks[i]

            im[:, :-shift] = im[:, shift:]
            target[:, :-shift] = target[:, shift:]
            images.append(im)
            masks.append(target)

        # shift up
        rand_n = list(np.random.randint(low=0, high=dataset_len, size=dataset_len // 4))
        for i in rand_n:
            shift = 20
            im = images[i]
            target = masks[i]

            im[:-shift, :] = im[shift:, :]
            target[:-shift, :] = target[shift:, :]
            images.append(im)
            masks.append(target)

        # shift down
        rand_n = list(np.random.randint(low=0, high=dataset_len, size=dataset_len // 4))
        for i in rand_n:
            shift = 20
            im = images[i]
            target = masks[i]

            im[shift:, :] = im[:-shift, :]
            target[shift:, :] = target[:-shift, :]
            images.append(im)
            masks.append(target)
    else:
        for i_i_fl, img_fl in enumerate(tqdm(images_files)):
            images.append(np.array(
                Image.open(img_fl)
            ))
            masks.append(np.array(
                Image.open(masks_files[i_i_fl])
            ))

    return images, masks
