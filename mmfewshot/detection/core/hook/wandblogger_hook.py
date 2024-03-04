import os.path as osp

from mmcv.runner import HOOKS
from mmcv.runner.dist_utils import master_only
from mmdet.core.hook.wandblogger_hook import MMDetWandbHook


@HOOKS.register_module()
class MMfewshotWandbHook(MMDetWandbHook):

    def after_val_epoch(self, runner) -> None:
        # TODO: implement mmfewshot wandb hook in epoch-based runner
        return super().after_val_epoch(runner)

    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner)
        if tags:
            if self.with_step:
                if self.get_mode(runner) == 'val':
                    self.wandb.log(tags, commit=self.commit)
                else:
                    self.wandb.log(
                        tags, step=self.get_iter(runner), commit=self.commit)
            else:
                tags['global_step'] = self.get_iter(runner)
                self.wandb.log(tags, commit=self.commit)
