import argparse
import copy
import math
import os
import parser

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from tqdm import tqdm

import mmfewshot
from mmfewshot.detection.datasets import build_dataloader, build_dataset


def CropResizeInstance(img, gt_bbox, gt_label, num_context_pixels,
                       target_size):
    """Crop and resize instance from image.

    Args:
        img (_type_): raw image
        gt_bbox (_type_): ground truth bounding box
        gt_label (_type_): ground truth label
        num_context_pixels (_type_): number of context pixels
        target_size (_type_): target size eg (224, 224)

    Returns:
        _type_: dict with keys: img, gt_bbox, gt_label
    """
    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = list(map(int, gt_bbox.tolist()))
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    t_x1, t_y1, t_x2, t_y2 = 0, 0, bbox_w, bbox_h
    if bbox_h <= target_size[0] / 4 or bbox_w <= target_size[0] / 4:
        return None

    if bbox_w >= bbox_h:
        crop_x1 = x1 - num_context_pixels
        crop_x2 = x2 + num_context_pixels
        t_x1 = t_x1 + num_context_pixels
        t_x2 = t_x1 + bbox_w
        if crop_x1 < 0:
            t_x1 = t_x1 + crop_x1
            t_x2 = t_x1 + bbox_w
            crop_x1 = 0
        if crop_x2 > img_w:
            crop_x2 = img_w

        short_size = bbox_h
        long_size = crop_x2 - crop_x1
        y_center = int((y2 + y1) / 2)  # math.ceil((y2 + y1) / 2)
        crop_y1 = int(
            y_center -
            (long_size / 2))  # int(y_center - math.ceil(long_size / 2))
        crop_y2 = int(
            y_center +
            (long_size / 2))  # int(y_center + math.floor(long_size / 2))

        # t_y1 and t_y2 will change when crop context or overflow
        t_y1 = t_y1 + math.ceil((long_size - short_size) / 2)
        t_y2 = t_y1 + bbox_h

        if crop_y1 < 0:
            t_y1 = t_y1 + crop_y1
            t_y2 = t_y1 + bbox_h
            crop_y1 = 0
        if crop_y2 > img_h:
            crop_y2 = img_h

        crop_short_size = crop_y2 - crop_y1
        crop_long_size = crop_x2 - crop_x1
        square = np.zeros((crop_long_size, crop_long_size, 3), dtype=np.uint8)
        delta = int(
            (crop_long_size - crop_short_size) /
            2)  # int(math.ceil((crop_long_size - crop_short_size) / 2))

        square_y1 = delta
        square_y2 = delta + crop_short_size

        t_y1 = t_y1 + delta
        t_y2 = t_y2 + delta

        crop_box = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
        square[square_y1:square_y2, :, :] = crop_box
    else:
        crop_y1 = y1 - num_context_pixels
        crop_y2 = y2 + num_context_pixels

        # t_y1 and t_y2 will change when crop context or overflow
        t_y1 = t_y1 + num_context_pixels
        t_y2 = t_y1 + bbox_h
        if crop_y1 < 0:
            t_y1 = t_y1 + crop_y1
            t_y2 = t_y1 + bbox_h
            crop_y1 = 0
        if crop_y2 > img_h:
            crop_y2 = img_h

        short_size = bbox_w
        long_size = crop_y2 - crop_y1
        x_center = int((x2 + x1) / 2)  # math.ceil((x2 + x1) / 2)
        crop_x1 = int(
            x_center -
            (long_size / 2))  # int(x_center - math.ceil(long_size / 2))
        crop_x2 = int(
            x_center +
            (long_size / 2))  # int(x_center + math.floor(long_size / 2))

        # t_x1 and t_x2 will change when crop context or overflow
        t_x1 = t_x1 + math.ceil((long_size - short_size) / 2)
        t_x2 = t_x1 + bbox_w
        if crop_x1 < 0:
            t_x1 = t_x1 + crop_x1
            t_x2 = t_x1 + bbox_w
            crop_x1 = 0
        if crop_x2 > img_w:
            crop_x2 = img_w

        crop_short_size = crop_x2 - crop_x1
        crop_long_size = crop_y2 - crop_y1
        square = np.zeros((crop_long_size, crop_long_size, 3), dtype=np.uint8)
        delta = int(
            (crop_long_size - crop_short_size) /
            2)  # int(math.ceil((crop_long_size - crop_short_size) / 2))
        square_x1 = delta
        square_x2 = delta + crop_short_size

        t_x1 = t_x1 + delta
        t_x2 = t_x2 + delta
        crop_box = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
        square[:, square_x1:square_x2, :] = crop_box

    square = square.astype(np.float32, copy=False)
    square, square_scale = mmcv.imrescale(
        square, target_size, return_scale=True, backend='cv2')
    square = square.astype(np.uint8)
    t_x1 = int(t_x1 * square_scale)
    t_y1 = int(t_y1 * square_scale)
    t_x2 = int(t_x2 * square_scale)
    t_y2 = int(t_y2 * square_scale)

    results = dict()
    results['img'] = square
    results['img_shape'] = square.shape
    results['gt_bboxes'] = np.array([[t_x1, t_y1, t_x2,
                                      t_y2]]).astype(np.float32)
    results['gt_labels'] = gt_label.cpu().numpy()
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate diffusion data from mmfewshot dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('save_dir', help='the dir to save generated data')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # build train dataloader
    dataset = build_dataset(cfg.data.generation)
    train_dataloader_default_args = dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        num_gpus=1,
        dist=False,
        seed=42,
        data_cfg=copy.deepcopy(cfg.data),
        persistent_workers=False)
    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }
    data_loader = build_dataloader(dataset, **train_loader_cfg)
    CLASSES = data_loader.dataset.CLASSES
    for i, data_batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        img = data_batch['img'].data[0][0].cpu().numpy().transpose(
            1, 2, 0).astype('uint8')
        for j in range(data_batch['gt_bboxes'].data[0][0].size(0)):
            # crop and resize
            # temp_img = copy.deepcopy(img) # to avoid resize raw img
            result = CropResizeInstance(img,
                                        data_batch['gt_bboxes'].data[0][0][j],
                                        data_batch['gt_labels'].data[0][0][j],
                                        4, (512, 512))
            if result is not None:
                save_img = result['img']
                save_label = result['gt_labels']
                save_class_name = CLASSES[save_label]
                if not os.path.exists(
                        os.path.join(args.save_dir, save_class_name)):
                    os.mkdir(os.path.join(args.save_dir, save_class_name))
                cv2.imwrite(
                    os.path.join(args.save_dir, save_class_name,
                                 str(i) + '_' + str(j) + '.jpg'), save_img)
