# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmfewshot.detection.datasets import (build_dataloader, build_dataset,
                                          get_copy_dataset_type)
from mmfewshot.detection.models import build_detector
from mmfewshot.utils import compat_cfg
from mmfewshot.utils.parse_args import parse_test_args


def test():

    assert args.out or args.eval or args.show \
        or args.show_dir, (
            'Please specify at least one operation (save/eval/show the '
            'results / save the results) with the argument "--out", "--eval"',
            '"--show" or "--show-dir"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    cfg = compat_cfg(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)
    # update overall dataloader(for train, val and test) setting
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # currently only support single images testing
    assert test_loader_cfg['samples_per_gpu'] == 1, \
        'currently only support single images testing'
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # pop frozen_parameters
    cfg.model.pop('frozen_parameters', None)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    # for meta-learning methods which require support template dataset
    # for model initialization.
    if cfg.data.get('model_init', None) is not None:
        cfg.data.model_init.pop('copy_from_train_dataset')
        model_init_samples_per_gpu = cfg.data.model_init.pop(
            'samples_per_gpu', 1)
        model_init_workers_per_gpu = cfg.data.model_init.pop(
            'workers_per_gpu', 1)
        if cfg.data.model_init.get('ann_cfg', None) is None:
            assert checkpoint['meta'].get('model_init_ann_cfg',
                                          None) is not None
            cfg.data.model_init.type = \
                get_copy_dataset_type(cfg.data.model_init.type)
            cfg.data.model_init.ann_cfg = \
                checkpoint['meta']['model_init_ann_cfg']
        model_init_dataset = build_dataset(cfg.data.model_init)
        # disable dist to make all rank get same data
        model_init_dataloader = build_dataloader(
            model_init_dataset,
            samples_per_gpu=model_init_samples_per_gpu,
            workers_per_gpu=model_init_workers_per_gpu,
            dist=False,
            shuffle=False)
    if args.input:
        outputs = mmcv.load(args.input)
    else:
        if not distributed:
            # Please use MMCV >= 1.4.4 for CPU testing!
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)
            show_kwargs = dict(show_score_thr=args.show_score_thr)
            if cfg.data.get('model_init', None) is not None:
                from mmfewshot.detection.apis import (single_gpu_model_init,
                                                    single_gpu_test)
                single_gpu_model_init(model, model_init_dataloader)
            else:
                from mmdet.apis.test import single_gpu_test
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                    **show_kwargs)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            if cfg.data.get('model_init', None) is not None:
                from mmfewshot.detection.apis import (multi_gpu_model_init,
                                                    multi_gpu_test)
                multi_gpu_model_init(model, model_init_dataloader)
            else:
                from mmdet.apis.test import multi_gpu_test
            outputs = multi_gpu_test(
                model,
                data_loader,
                args.tmpdir,
                args.gpu_collect,
            )

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    args = parse_test_args()
    test()
