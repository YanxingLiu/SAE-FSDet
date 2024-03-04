import os
import os.path as osp
from typing import Dict, List, Optional, Sequence, Union

import mmcv
import numpy as np
from mmcls.datasets.builder import DATASETS
from typing_extensions import Literal

from .base import BaseFewShotDataset

TRAIN_CLASSES = ['2S1', 'BMP2', 'BRDM2', 'BTR60', 'BTR70', 'D7', 'T62']
TEST_CLASSES = ['T72', 'ZIL131', 'ZSU234']


@DATASETS.register_module()
class MStarDataset(BaseFewShotDataset):
    TRAIN_CLASSES = TRAIN_CLASSES
    TEST_CLASSES = TEST_CLASSES

    def __init__(self,
                 subset: Literal['train', 'test', 'val'] = 'train',
                 file_format: str = 'tif',
                 *args,
                 **kwargs) -> None:
        if isinstance(subset, str):
            subset = [subset]
        for subset_ in subset:
            assert subset_ in ['train', 'test']
        self.subset = subset
        self.file_format = file_format
        super().__init__(*args, **kwargs)

    def get_classes(
            self,
            classes: Optional[Union[Sequence[str],
                                    str]] = None) -> Sequence[str]:
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): Three types of input
                will correspond to different processing logics:

                - If `classes` is a tuple or list, it will override the
                  CLASSES predefined in the dataset.
                - If `classes` is None, we directly use pre-defined CLASSES
                  will be used by the dataset.
                - If `classes` is a string, it is the path of a classes file
                  that contains the name of all classes. Each line of the file
                  contains a single class name.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            class_names = []
            for subset_ in self.subset:
                if subset_ == 'train':
                    class_names += self.TRAIN_CLASSES
                elif subset_ == 'test':
                    class_names += self.TEST_CLASSES
                else:
                    raise ValueError(f'invalid subset {subset_} only '
                                     f'support train, val or test.')
        elif isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names

    def load_annotations(self) -> List:
        """Load annotation according to the classes subset."""
        if self.subset[0] == 'train':
            image_prefix = '17DEG'
        elif self.subset[0] == 'test':
            image_prefix = '15DEG'
        else:
            raise ValueError(f'invalid subset {self.subset} only '
                             f'support train or test.')
        img_file_list = {
            class_name: sorted(
                os.listdir(
                    osp.join(self.data_prefix, image_prefix, class_name)))
            for class_name in self.CLASSES
        }
        data_infos = []
        for _, class_name in enumerate(self.CLASSES):
            for img_name in img_file_list[class_name]:
                gt_label = self.class_to_idx[class_name]
                info = {
                    'img_prefix':
                    osp.join(self.data_prefix, image_prefix, class_name),
                    'img_info': {
                        'filename': img_name,
                    },
                    'gt_label':
                    np.array(gt_label, dtype=np.int64)
                }
                data_infos.append(info)
        return data_infos
