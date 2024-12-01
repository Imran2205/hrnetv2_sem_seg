# Semantic segmentation experiments using the annotated samples from UWSv2 dataset
train and test using HighResolutionNet (HRNet) model

## Train
!python train_hrnet_v2_uws.py --cfg ../experiments/hrnet/uws_v2_train_hrnet_v2_WACV25_CAMERA_READY.yaml

## Test
!python test_hrnet_v2_uws.py --cfg ../experiments/hrnet/uws_v2_test_hrnet_v2_WACV25_CAMERA_READY.yaml
