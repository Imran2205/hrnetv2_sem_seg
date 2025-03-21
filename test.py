import argparse
import os
import pprint

import logging
import timeit

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

from networks import hrnet_v2 as models

from core.criterion import CrossEntropy, OhemCrossEntropy
from core.function import validate
from networks.hrnet_v2.config import config_hrnet_v2 as config
from networks.hrnet_v2.config import update_config_hrnet_v2 as update_config
from networks.hrnet_v2.hrnet_v2_utils.utils import create_logger, FullModel
from networks.hrnet_v2.hrnet_v2_utils.normalization_utils import get_imagenet_mean_std
from dataloaders.uws.uws_full_dataloader import UWSDataLoader
from networks.hrnet_v2.hrnet_v2_utils import transform
from utils.load_images_and_masks import load_images_and_masks
from utils.print_iou_clean import print_selected_iou


def parse_args():
    """
        Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    """
        Test HRNet V2 network on validation set of Underwater Segmentation dataset
    """
    args = parse_args()

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)

    batch_size = config.TEST.BATCH_SIZE_PER_GPU

    # prepare data
    mean, std = get_imagenet_mean_std()

    if config.DATASET.DATASET == 'UWS':
        val_transform_list = [
            transform.ResizeShort(config.TRAIN.IMAGE_SIZE[0]),
            transform.Crop(
                [config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1]],
                crop_type="center",
                padding=mean,
                ignore_label=config.TRAIN.IGNORE_LABEL,
            ),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std),
        ]

        val_dir = config.DATASET.TEST_SET
        images_test, masks_test = load_images_and_masks(config.DATASET.ROOT, val_dir, augment=False)

        val_dataset = UWSDataLoader(
            output_image_height=config.TRAIN.IMAGE_SIZE[0],
            images=images_test,
            masks=masks_test,
            transform=transform.Compose(val_transform_list),
            channel_values=None
        )
    else:
        val_dataset = None
        logger.info("=> no dataset found. " 'Exiting...')
        exit()

    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=True
    )
    logger.info(f'Validation loader has len: {len(val_loader)}')

    if torch.__version__.startswith('1'):
        module = eval('models.' + config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d

    model = eval('models.' + config.MODEL.NAME +
                 '.get_seg_model')(config)

    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP)  # ,weight=train_dataset.class_weights)
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL)  # ,weight=train_dataset.class_weights)

    model = FullModel(model, criterion)

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':

        params_dict = dict(model.named_parameters())
        if config.TRAIN.NONBACKBONE_KEYWORDS:
            bb_lr = []
            nbb_lr = []
            nbb_keys = set()
            for k, param in params_dict.items():
                if any(part in k for part in config.TRAIN.NONBACKBONE_KEYWORDS):
                    nbb_lr.append(param)
                    nbb_keys.add(k)
                else:
                    bb_lr.append(param)
            print(nbb_keys)
            params = [{'params': bb_lr, 'lr': config.TRAIN.LR},
                      {'params': nbb_lr, 'lr': config.TRAIN.LR * config.TRAIN.NONBACKBONE_MULT}]
        else:
            params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(params,
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV,
                                    )
    else:
        raise ValueError('Only Support SGD optimizer')

    if config.TEST.MODEL_FILE and config.TEST.DATA_PARALLEL:
        gpus = list(config.GPUS)
        model = nn.DataParallel(model, device_ids=gpus).cuda()
        model_state_file = config.TEST.MODEL_FILE
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            model.module.load_state_dict(checkpoint)
            logger.info("=> Using Data Parallel")
            logger.info("=> loaded pretrained model {}"
                        .format(config.MODEL.PRETRAINED))
    elif not config.TEST.DATA_PARALLEL:
        if config.TEST.MODEL_FILE:
            model_state_file = config.TEST.MODEL_FILE
        else:
            model_state_file = os.path.join(final_output_dir, 'final_state.pth')
        logger.info("=> Not using Data Parallel")
        logger.info('=> loading model from {}'.format(model_state_file))

        pretrained_dict = torch.load(model_state_file)
        model.load_state_dict(pretrained_dict)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    else:
        raise ValueError('Could not load the model.')

    start = timeit.default_timer()

    valid_loss, _, IoU_array = validate(config,
                                               val_loader, model, writer_dict)

    mean_IoU = print_selected_iou(
        iou_list=IoU_array,
        label_dict=val_dataset.get_label_dict(),
        selected_ids=[9, 10] + list(range(21, 29)) + list(range(30, 51)),
        logger=logger
    )

    msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}'.format(
        valid_loss, mean_IoU)
    logging.info(msg)
    logging.info(IoU_array)

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % int((end - start) / 3600))
    logger.info('Done')


if __name__ == '__main__':
    main()
