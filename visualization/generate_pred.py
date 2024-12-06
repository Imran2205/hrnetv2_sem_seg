import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import torch
# for composing the transform for both 'image' as well as 'label'
from utils.hrnet_utils import transform
# import torchvision.transforms as transforms # HRNet doesn't do basic torchvision's transform which takes only one tensor (either the image or the label)

from utils.hrnet_v2_utils.utils import FullModel
from networks.hrnet_v2.seg_hrnet import HighResolutionNet
from core.criterion import CrossEntropy, OhemCrossEntropy
import torch.nn as nn
from torch.nn import functional as F
import os


def create_color_map(segmentation_label_dictionary):
    """Create a mapping from train_id to RGB color values."""
    color_map = {}
    for item in segmentation_label_dictionary.values():
        train_id = item['train_id']
        color = item['color']
        color_map[train_id] = color
    return color_map


def save_colored_prediction(pred, save_path, color_map):
    """Convert prediction to RGB image using the color map and save it."""
    height, width = pred.shape
    colored_pred = np.zeros((height, width, 3), dtype=np.uint8)

    for train_id, color in color_map.items():
        mask = pred == train_id
        colored_pred[mask] = color

    img = Image.fromarray(colored_pred)
    img.save(save_path)


def process_images(input_dir, output_dir, model, preprocess, device, color_map):
    """Process all images in input directory and save predictions."""
    # Create output directories if they don't exist
    os.makedirs(os.path.join(output_dir, 'raw_predictions'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'colored_predictions'), exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    model.eval()
    with torch.no_grad():
        for image_file in tqdm(image_files, desc="Processing images"):
            # Load and preprocess image
            img_path = os.path.join(input_dir, image_file)
            img = Image.open(img_path).convert('RGB')

            # Create dummy label for the preprocess function
            dummy_label = np.zeros((img.height, img.width), dtype=np.uint8)
            img_tensor, _ = preprocess(np.array(img), dummy_label)

            # Prepare input for model
            image = img_tensor.unsqueeze(0).to(device)

            # Forward pass
            pred = model.model(image)

            # Interpolate prediction to original image size
            pred = F.interpolate(
                pred, (img.height, img.width),
                mode='bilinear', align_corners=True
            )

            # Convert prediction to numpy array
            output = pred.cpu().detach().numpy().transpose(0, 2, 3, 1)
            output = output.squeeze(0)
            seg_pred = np.argmax(output, axis=2)

            # Save raw prediction
            raw_pred_path = os.path.join(output_dir, 'raw_predictions', f'{os.path.splitext(image_file)[0]}_pred.npy')
            np.save(raw_pred_path, seg_pred)

            # Save colored prediction
            colored_pred_path = os.path.join(output_dir, 'colored_predictions',
                                             f'{os.path.splitext(image_file)[0]}_colored.png')
            save_colored_prediction(seg_pred, colored_pred_path, color_map)


# Main execution
if __name__ == "__main__":
    import argparse
    import yaml


    class DotDict(dict):
        """Custom dictionary supporting dot notation access."""

        def __getattr__(self, item):
            value = self.get(item)
            if isinstance(value, dict):
                return DotDict(value)  # recursively convert nested dictionaries
            return value

        def __setattr__(self, key, value):
            self[key] = value

        def __delattr__(self, key):
            del self[key]


    # ------------------------------------------------------------------
    from config import \
        config_hrnet_v2 as config  # from config directory ('package' more precisely) loading 'default_hrnet_v2.py' as 'config_hrnet_v2' which is subsequently imported as 'config'
    from config import update_config_hrnet_v2 as update_config

    # ------------------------------------------------------------------

    config_file_name = "/content/hrnetv2_sem_seg/experiments/hrnet/uws_training_hrnet_v2_test.yaml"

    try:
        with open(config_file_name) as file:
            config = yaml.safe_load(file)
            print(f"YAML file content loaded: {config}")
    except FileNotFoundError:
        print(f"Error: File '{config_file_name}' not found.")

    print(f"checking the values as dictionary keys: {config['CUDNN']}")
    cfg = DotDict(config)
    print(f"checking the values as dictionary keys: {cfg.CUDNN}")
    print(f"checking the values as dictionary keys: {cfg.CUDNN.BENCHMARK}")
    print(f"checking the values as dictionary keys: {cfg.MODEL.PRETRAINED}")
    print(f"checking the values as dictionary keys: {cfg.LOSS.OHEMTHRES}")

    segmentation_label_dictionary = {
        0: {'name': 'unlabeled', 'train_id': 255, 'color': (0, 0, 0)},
        1: {'name': 'crab', 'train_id': 0, 'color': (128, 64, 128)},
        2: {'name': 'crocodile', 'train_id': 1, 'color': (244, 35, 232)},
        3: {'name': 'dolphin', 'train_id': 2, 'color': (70, 70, 70)},
        4: {'name': 'frog', 'train_id': 3, 'color': (102, 102, 156)},
        5: {'name': 'nettles', 'train_id': 4, 'color': (190, 153, 153)},
        6: {'name': 'octopus', 'train_id': 5, 'color': (153, 153, 153)},
        7: {'name': 'otter', 'train_id': 6, 'color': (250, 170, 30)},
        8: {'name': 'penguin', 'train_id': 7, 'color': (220, 220, 0)},
        9: {'name': 'polar_bear', 'train_id': 8, 'color': (107, 142, 35)},
        10: {'name': 'sea_anemone', 'train_id': 9, 'color': (152, 251, 152)},
        11: {'name': 'sea_urchin', 'train_id': 10, 'color': (70, 130, 180)},
        12: {'name': 'seahorse', 'train_id': 11, 'color': (220, 20, 60)},
        13: {'name': 'seal', 'train_id': 12, 'color': (253, 0, 0)},
        14: {'name': 'shark', 'train_id': 13, 'color': (0, 0, 142)},
        15: {'name': 'shrimp', 'train_id': 14, 'color': (0, 0, 70)},
        16: {'name': 'star_fish', 'train_id': 15, 'color': (0, 60, 100)},
        17: {'name': 'stingray', 'train_id': 16, 'color': (0, 80, 100)},
        18: {'name': 'squid', 'train_id': 17, 'color': (0, 0, 230)},
        19: {'name': 'turtle', 'train_id': 18, 'color': (119, 11, 32)},
        20: {'name': 'whale', 'train_id': 19, 'color': (111, 74, 0)},
        21: {'name': 'nudibranch', 'train_id': 20, 'color': (81, 0, 81)},
        22: {'name': 'coral', 'train_id': 21, 'color': (250, 170, 160)},
        23: {'name': 'rock', 'train_id': 22, 'color': (230, 150, 140)},
        24: {'name': 'water', 'train_id': 23, 'color': (180, 165, 180)},
        25: {'name': 'sand', 'train_id': 24, 'color': (150, 100, 100)},
        26: {'name': 'plant', 'train_id': 25, 'color': (150, 120, 90)},
        27: {'name': 'human', 'train_id': 26, 'color': (153, 153, 153)},
        28: {'name': 'reef', 'train_id': 27, 'color': (0, 0, 110)},
        29: {'name': 'others', 'train_id': 28, 'color': (47, 220, 70)}
    }


    def get_imagenet_mean_std() -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """ See use here in Pytorch ImageNet script:
            https://github.com/pytorch/examples/blob/master/imagenet/main.py#L197

            Returns:
            -   mean: Tuple[float,float,float],
            -   std: Tuple[float,float,float] = None
        """
        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]
        return mean, std


    # util function:
    mean, std = get_imagenet_mean_std()

    preprocess = custom_transform.Compose(
        [
            custom_transform.ResizeShort(cfg.TRAIN.IMAGE_SIZE[0]),
            custom_transform.Crop(
                [cfg.TRAIN.IMAGE_SIZE[0], cfg.TRAIN.IMAGE_SIZE[1]],
                crop_type="center",
                padding=mean,
                ignore_label=cfg.TRAIN.IGNORE_LABEL,
            ),
            custom_transform.ToTensor(),
            custom_transform.Normalize(mean=mean, std=std),
        ]
    )

    model = HighResolutionNet(cfg)
    model.init_weights(cfg.MODEL.PRETRAINED)
    criterion = CrossEntropy(ignore_label=cfg.TRAIN.IGNORE_LABEL)  # ,weight=train_dataset.class_weights)
    # criterion = OhemCrossEntropy(ignore_label=cfg.TRAIN.IGNORE_LABEL, thres=cfg.LOSS.OHEMTHRES, min_kept=cfg.LOSS.OHEMKEEP)  # ,weight=train_dataset.class_weights)

    model = FullModel(model, criterion)
    # gpus = list([0])
    # model = nn.DataParallel(model, device_ids=gpus).cuda()

    finetuned_model_state_file = "/content/hrnetv2_sem_seg/output/best.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(finetuned_model_state_file, map_location={'cuda:0': 'cpu'})

    model.load_state_dict(checkpoint)
    model.to(device)
    # Input and output directories
    input_dir = '/home/ibk5106/projects/projects/uws/uwss_v2/validation/images'
    output_dir = '/home/ibk5106/projects/projects/uws/model_pred'

    # Create color map from the label dictionary
    color_map = create_color_map(segmentation_label_dictionary)

    # Process all images
    process_images(
        input_dir=input_dir,
        output_dir=output_dir,
        model=model,
        preprocess=preprocess,
        device=device,
        color_map=color_map
    )