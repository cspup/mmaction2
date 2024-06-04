#!/usr/bin/env bash

python tools/visualizations/vis_cam.py \
  configs/recognition/lmnet/lmnet_imagenet-pretrained-r50_1xb32-1x1x8-25e_egogesture-rgb.py \
  autodl-tmp/mmaction2/work_dirs/lmnet_imagenet-pretrained-r50_1xb32-1x1x8-25e_egogesture-rgb/best_acc_top1_epoch_23.pth \
  /root/autodl-tmp/RS_v1_50/img/Subject50-Scene1-rgb1-18-1090-1130/ \
  --use-frames \
  --target-layer-name backbone/layer4/1/block/relu \
  --out-filename lmnet_cam.mp4 \
  --fps 8