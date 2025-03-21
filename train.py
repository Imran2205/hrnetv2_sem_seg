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

from networks.hrnet_v2.config import config_hrnet_v2 as config
from networks.hrnet_v2.config import update_config_hrnet_v2 as update_config
from core.criterion import CrossEntropy, OhemCrossEntropy
from core.function import train, validate
from networks.hrnet_v2.hrnet_v2_utils.utils import create_logger, FullModel
from networks.hrnet_v2.hrnet_v2_utils.normalization_utils import get_imagenet_mean_std
from utils.load_images_and_masks import load_images_and_masks
from dataloaders.uws.uws_full_dataloader import UWSDataLoader
from networks.hrnet_v2.hrnet_v2_utils import transform
from utils.print_iou_clean import print_selected_iou


def parse_args():
    """
        Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train segmentation network')

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
    Train HRNet V2 network on Underwater Segmentation dataset
    - UWSv1
    - UWSv2
    """
    args = parse_args()

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

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

    model = eval('models.' + config.MODEL.NAME +
                 '.get_seg_model')(config)

    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU

    # prepare data
    mean, std = get_imagenet_mean_std()
    
    if config.DATASET.DATASET == 'UWS':
        train_transform_list = [
            transform.ResizeShort(config.TRAIN.IMAGE_SIZE[0]),
            transform.Crop(
                [config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1]],
                crop_type="rand",
                padding=mean,
                ignore_label=config.TRAIN.IGNORE_LABEL,
            ),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std),
        ]
        # transform.ResizeTest((config.TRAIN.TRAIN_H, config.TRAIN.TRAIN_W)),
        # transform.ResizeShort(config.TRAIN.SHORT_SIZE),
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
        
        train_dir = config.DATASET.TRAIN_SET
        val_dir = config.DATASET.TEST_SET

        images_train, masks_train = load_images_and_masks(config.DATASET.ROOT, train_dir, augment=True)
        images_test, masks_test = load_images_and_masks(config.DATASET.ROOT, val_dir, augment=False)

        '''
        import pdb
        pdb.set_trace()
        '''
        
        custom_transform_train = transform.Compose(train_transform_list)
        train_dataset = UWSDataLoader(
            output_image_height=config.TRAIN.IMAGE_SIZE[0],
            images=images_train,
            masks=masks_train,
            transform=custom_transform_train,
            channel_values=None
        )
        
        custom_transform_test = transform.Compose(val_transform_list)
        val_dataset = UWSDataLoader(
            output_image_height=config.TRAIN.IMAGE_SIZE[0],
            images=images_test,
            masks=masks_test,
            transform=custom_transform_test,
            channel_values=None
        )
    else:
        train_dataset = None
        val_dataset = None
        logger.info("=> no dataset found. " 'Exiting...')
        exit()

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    logger.info(f'Train loader has len: {len(train_loader)}')

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

    # criterion
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP)  # ,weight=train_dataset.class_weights)
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL)  # ,weight=train_dataset.class_weights)

    model = FullModel(model, criterion)

    model = nn.DataParallel(model, device_ids=gpus).cuda()
    logger.info(f'Using DataParallel')

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

    epoch_iters = int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU)  # / len(gpus)

    best_mIoU = 0
    best_epoch = 0
    last_epoch = 0
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            if 'best_epoch' in checkpoint.keys():
                best_epoch = checkpoint['best_epoch']
            last_epoch = checkpoint['epoch']

            model.module.model.load_state_dict(
                {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))

    extra_epoch_iters = 0
    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    extra_iters = config.TRAIN.EXTRA_EPOCH * extra_epoch_iters

    for epoch in range(last_epoch, end_epoch):

        current_trainloader = train_loader if epoch >= config.TRAIN.END_EPOCH else train_loader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)

        # valid_loss, mean_IoU, IoU_array = validate(config,
        #             testloader, model, writer_dict)

        if epoch >= config.TRAIN.END_EPOCH:
            train(config, epoch - config.TRAIN.END_EPOCH,
                  config.TRAIN.EXTRA_EPOCH, extra_epoch_iters,
                  config.TRAIN.EXTRA_LR, extra_iters,
                  train_loader, optimizer, model, writer_dict)
        else:
            train(config, epoch, config.TRAIN.END_EPOCH,
                  epoch_iters, config.TRAIN.LR, num_iters,
                  train_loader, optimizer, model, writer_dict)

        valid_loss, _, IoU_array = validate(config, val_loader, model, writer_dict)

        mean_IoU = print_selected_iou(
            iou_list=IoU_array,
            label_dict=train_dataset.get_label_dict(),
            selected_ids=[9, 10] + list(range(21, 29)) + list(range(30, 51)),
            logger=logger
        )

        if mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
            best_epoch = epoch
            torch.save(model.module.state_dict(),
                       os.path.join(final_output_dir, 'best.pth'))

        logger.info('=> saving checkpoint to {}'.format(
            final_output_dir + 'checkpoint.pth.tar'))
        torch.save({
            'epoch': epoch + 1,
            'best_mIoU': best_mIoU,
            'best_epoch': best_epoch,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))

        msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}, Best_epoch: {}'.format(
            valid_loss, mean_IoU, best_mIoU, best_epoch)
        logger.info(msg)
        logging.info(IoU_array)

    torch.save(model.module.state_dict(),
               os.path.join(final_output_dir, 'final_state.pth'))

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % int((end - start) / 3600))
    logger.info('Done')


if __name__ == '__main__':
    main()


## separation between epoch
## epoch + 1 during print
