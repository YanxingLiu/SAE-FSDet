# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
import time

import torch
from mmcv.runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..detection.models.utils.checkpoint_vitae import save_checkpoint

try:
    import apex
except:
    print('apex is not installed')


@RUNNERS.register_module()
class InfiniteEpochBasedRunner(EpochBasedRunner):
    """Epoch-based Runner supports dataloader with InfiniteSampler.

    The workers of dataloader will re-initialize, when the iterator of
    dataloader is created. InfiniteSampler is designed to avoid these time
    consuming operations, since the iterator with InfiniteSampler will never
    reach the end.
    """

    def train(self, data_loader: DataLoader, **kwargs) -> None:
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        # To reuse the iterator, we only create iterator once and bind it
        # with runner. In the next epoch, the iterator will be used against
        if not hasattr(self, 'data_loader_iter'):
            self.data_loader_iter = iter(self.data_loader)

        # The InfiniteSampler will never reach the end, but we set the
        # length of InfiniteSampler to the actual length of dataset.
        # The length of dataloader is determined by the length of sampler,
        # when the sampler is not None. Therefore, we can simply forward the
        # whole dataset in a epoch by length of dataloader.

        for i in range(len(self.data_loader)):
            data_batch = next(self.data_loader_iter)
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1


@RUNNERS.register_module()
class EpochBasedRunnerAmp(EpochBasedRunner):
    """Epoch-based Runner with AMP support.

    This runner train models epoch by epoch.
    """

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            # if platform.system() != 'Windows':
            #     mmcv.symlink(filename, dst_file)
            # else:
            shutil.copy(filepath, dst_file)

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                checkpoint = self.load_checkpoint(checkpoint)
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k])
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        if 'amp' in checkpoint:
            apex.amp.load_state_dict(checkpoint['amp'])
            self.logger.info('load amp state dict')

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)
