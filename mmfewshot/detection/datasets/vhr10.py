import os.path as osp
import re
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets import CustomDataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
from PIL import Image
from terminaltables import AsciiTable

from mmfewshot.detection.core import eval_map
from mmfewshot.detection.datasets import BaseFewShotDataset
from mmfewshot.utils import get_root_logger
from ..evaluation.recall import eval_recalls, print_per_class_recall

VHR10_SPLIT = dict(
    CLASSES=('airplane', 'ship', 'storage-tank', 'baseball-diamond',
             'tennis-court', 'basketball-court', 'ground-track-field',
             'harbor', 'bridge', 'vehicle'),
    BASE_CLASSES_SPLIT1=('ship', 'storage-tank', 'basketball-court',
                         'ground-track-field', 'harbor', 'bridge', 'vehicle'),
    NOVEL_CLASSES_SPLIT1=('airplane', 'baseball-diamond', 'tennis-court'),
    ALL_CLASSES_SPLIT1=('ship', 'storage-tank', 'basketball-court',
                        'ground-track-field', 'harbor', 'bridge', 'vehicle',
                        'airplane', 'baseball-diamond', 'tennis-court'),
    BASE_CLASSES_SPLIT2=('airplane', 'ship', 'storage-tank',
                         'baseball-diamond', 'tennis-court', 'harbor',
                         'bridge'),
    NOVEL_CLASSES_SPLIT2=('basketball-court', 'ground-track-field', 'vehicle'),
    ALL_CLASSES_SPLIT2=('airplane', 'ship', 'storage-tank', 'baseball-diamond',
                        'tennis-court', 'harbor', 'bridge', 'basketball-court',
                        'ground-track-field', 'vehicle'))


@DATASETS.register_module()
class VHR10Dataset(CustomDataset):
    CLASSES = ('airplane', 'ship', 'storage tank', 'baseball diamond',
               'tennis court', 'basketball court', 'ground track field',
               'harbor', 'bridge', 'vehicle')

    def __init__(self,
                 ann_file,
                 pipeline,
                 min_size=None,
                 img_subdir='JPEGImages',
                 ann_subdir='Annotations',
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 seg_suffix='.png',
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk')):

        self.img_subdir = img_subdir
        self.ann_subdir = ann_subdir
        self.min_size = min_size  # min size is used in filter too small annotations

        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.seg_suffix = seg_suffix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.file_client = mmcv.FileClient(**file_client_args)
        self.CLASSES = self.get_classes(classes)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)

        # load annotations (and proposals)
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations(local_path)
        else:
            warnings.warn(
                'The used MMCV version does not have get_local_path. '
                f'We treat the {self.ann_file} as local paths and it '
                'might cause errors if the path is not a local path. '
                'Please use MMCV>= 1.3.16 if you meet errors.')
            self.data_infos = self.load_annotations(self.ann_file)
        if self.proposal_file is not None:
            if hasattr(self.file_client, 'get_local_path'):
                with self.file_client.get_local_path(
                        self.proposal_file) as local_path:
                    self.proposals = self.load_proposals(local_path)
            else:
                warnings.warn(
                    'The used MMCV version does not have get_local_path. '
                    f'We treat the {self.ann_file} as local paths and it '
                    'might cause errors if the path is not a local path. '
                    'Please use MMCV>= 1.3.16 if you meet errors.')
                self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

        logger = get_root_logger()
        logger.info(self.__repr__())

    def __repr__(self) -> str:
        """Print the number of instances of each class."""
        result = (f'\n{self.__class__.__name__} '
                  f'with number of images {len(self)}, '
                  f'and instance counts: \n')
        if self.CLASSES is None:
            result += 'Category names are not provided. \n'
            return result
        instance_count = np.zeros(len(self.CLASSES) + 1).astype(int)
        # count the instance number in each image
        for idx in range(len(self)):
            label = self.get_ann_info(idx)['labels']
            unique, counts = np.unique(label, return_counts=True)
            if len(unique) > 0:
                # add the occurrence number to each class
                instance_count[unique] += counts
            else:
                # background is the last index
                instance_count[-1] += 1
        # create a table with category count
        table_data = [['category', 'count'] * 5]
        row_data = []
        for cls, count in enumerate(instance_count):
            if cls < len(self.CLASSES):
                row_data += [f'{cls} [{self.CLASSES[cls]}]', f'{count}']
            else:
                # add the background number
                row_data += ['-1 background', f'{count}']
            if len(row_data) == 10:
                table_data.append(row_data)
                row_data = []

        table = AsciiTable(table_data)
        result += table.table
        return result

    def _parse_txt_ann_info(self, txt_path):

        with open(txt_path) as f:
            annotations = f.readlines()
            f.close()

        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        # parase annotations
        for annotation in annotations:
            try:
                label = int(annotation.split(',')[-1]) - 1  # 注意这里在少样本的时候需要修改索引
            except:
                continue
            coordinates = re.findall(r'[(](.*?)[)]', annotation)
            x1, y1 = coordinates[0].split(',')
            x2, y2 = coordinates[1].split(',')
            bbox = np.array([float(x1), float(y1), float(x2), float(y2)])

            # ignore too small objs in training
            ignore = False
            if self.min_size is not None:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)

        #
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            # if use_coordinate
            # bboxes = np.array(bboxes, ndmin=2) - 1
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            # if use_coordinate
            # bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))

        return ann

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without annotation."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            if self.filter_empty_gt:
                ann = img_info['ann']
                if len(ann['labels']) == 0:
                    continue
                else:
                    valid_inds.append(i)
            else:
                valid_inds.append(i)

        return valid_inds

    def load_annotations(self, ann_file):
        """load annotations from txt.

        Args:
            ann_file (_type_): _description_
        """
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.label2cat = {i: cat for i, cat in enumerate(self.CLASSES)}
        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            file_name = osp.join(self.img_subdir, f'{img_id}.jpg')
            txt_path = osp.join(self.img_prefix, self.ann_subdir,
                                f'{img_id}.txt')
            img = Image.open(osp.join(self.img_prefix, file_name))
            width, height = img.size
            ann_info = self._parse_txt_ann_info(txt_path)
            data_infos.append(
                dict(
                    id=img_id,
                    filename=file_name,
                    width=width,
                    height=height,
                    ann=ann_info))

        return data_infos


@DATASETS.register_module()
class FewShotVHR10Dataset(BaseFewShotDataset):

    def __init__(self,
                 classes: Optional[Union[str, Sequence[str]]] = None,
                 num_novel_shots: Optional[int] = None,
                 num_base_shots: Optional[int] = None,
                 ann_shot_filter: Optional[Dict] = None,
                 use_difficult: bool = False,
                 min_bbox_area: Optional[Union[int, float]] = None,
                 dataset_name: Optional[str] = None,
                 test_mode: bool = False,
                 coordinate_offset: List[int] = [-1, -1, 0, 0],
                 img_subdir='positive image set',
                 ann_subdir='ground truth',
                 min_size=None,
                 **kwargs) -> None:
        if dataset_name is None:
            self.dataset_name = 'Test dataset' if test_mode else 'Train dataset'
        else:
            self.dataset_name = dataset_name

        self.SPLIT = VHR10_SPLIT
        self.split_id = None

        assert classes is not None, f'{self.dataset_name}: classes in ' \
                                    f'`FewShotVOCDataset` can not be None.'

        self.num_novel_shots = num_novel_shots
        self.num_base_shots = num_base_shots
        self.min_bbox_area = min_bbox_area
        self.classes = classes
        self.split_id = int(classes[-1])
        self.CLASSES = self.get_classes(classes)

        if ann_shot_filter is None:
            if num_novel_shots is not None or num_base_shots is not None:
                ann_shot_filter = self._create_ann_shot_filter()
        else:
            assert num_novel_shots is None and num_base_shots is None, \
                f'{self.dataset_name}: can not config ann_shot_filter and ' \
                f'num_novel_shots/num_base_shots at the same time.'
        self.coordinate_offset = coordinate_offset
        self.use_difficult = use_difficult
        self.img_subdir = img_subdir
        self.ann_subdir = ann_subdir
        self.min_size = min_size
        super().__init__(
            classes=None,
            ann_shot_filter=ann_shot_filter,
            dataset_name=dataset_name,
            test_mode=test_mode,
            **kwargs)

    def __repr__(self) -> str:
        """Print the number of instances of each class."""
        result = (f'\n{self.__class__.__name__} {self.dataset_name} '
                  f'with number of images {len(self)}, '
                  f'and instance counts: \n')
        if self.CLASSES is None:
            result += 'Category names are not provided. \n'
            return result
        instance_count = np.zeros(len(self.CLASSES) + 1).astype(int)
        # count the instance number in each image
        for idx in range(len(self)):
            label = self.get_ann_info(idx)['labels']
            unique, counts = np.unique(label, return_counts=True)
            if len(unique) > 0:
                # add the occurrence number to each class
                instance_count[unique] += counts
            else:
                # background is the last index
                instance_count[-1] += 1
        # create a table with category count
        table_data = [['category', 'count'] * 5]
        row_data = []
        for cls, count in enumerate(instance_count):
            if cls < len(self.CLASSES):
                row_data += [f'{cls} [{self.CLASSES[cls]}]', f'{count}']
            else:
                # add the background number
                row_data += ['-1 background', f'{count}']
            if len(row_data) == 10:
                table_data.append(row_data)
                row_data = []

        if row_data:
            extend_len = 10 - len(row_data)
            row_data.extend(extend_len * ['None'])
            table_data.append(row_data)

        table = AsciiTable(table_data)
        result += table.table
        return result

    def _parse_txt_ann_info(self, txt_path):

        with open(txt_path) as f:
            annotations = f.readlines()
            f.close()

        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        # parase annotations
        for annotation in annotations:
            if annotation == '\n':  # skip file ends
                continue
            ignore = False
            # label transform
            label = int(annotation.split(',')[-1]) - 1
            classes_name = self.SPLIT['CLASSES'][label]
            if classes_name not in self.CLASSES:
                ignore = True
                label = len(self.CLASSES)
            else:
                label = self.cat2label[classes_name]

            coordinates = re.findall(r'[(](.*?)[)]', annotation)
            x1, y1 = coordinates[0].split(',')
            x2, y2 = coordinates[1].split(',')
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            if not self.test_mode:
                bbox = [
                    i + offset
                    for i, offset in zip(bbox, self.coordinate_offset)
                ]

            # ignore too small objs in training
            if self.min_size is not None:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)

        #
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))

        return ann

    def _create_ann_shot_filter(self) -> Dict[str, int]:
        ann_shot_filter = {}
        if self.num_novel_shots is not None:
            for class_name in self.SPLIT[
                    f'NOVEL_CLASSES_SPLIT{self.split_id}']:
                ann_shot_filter[class_name] = self.num_novel_shots
        if self.num_base_shots is not None:
            for class_name in self.SPLIT[f'BASE_CLASSES_SPLIT{self.split_id}']:
                ann_shot_filter[class_name] = self.num_base_shots

        return ann_shot_filter

    def _filter_imgs(self,
                     min_size: int = 0,
                     min_bbox_area: Optional[int] = None) -> List[int]:
        valid_inds = []
        if min_bbox_area is None:
            min_bbox_area = self.min_bbox_area
        for i, img_info in enumerate(self.data_infos):
            # filter empty image
            if self.filter_empty_gt:
                cat_ids = img_info['ann']['labels'].astype(np.int64).tolist()
                if len(cat_ids) == 0:
                    continue
            # filter images smaller than `min_size`
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            # filter image with bbox smaller than min_bbox_area
            # it is usually used in Attention RPN
            if min_bbox_area is not None:
                skip_flag = False
                for bbox in img_info['ann']['bboxes']:
                    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if bbox_area < min_bbox_area:
                        skip_flag = True
                if skip_flag:
                    continue
            valid_inds.append(i)
        return valid_inds

    def _select_annotations(self, data_infos: List[Dict],
                            ann_shot_filter: Dict) -> List[Dict]:
        """Filter out extra annotations of specific class, while annotations of
        classes not in filter remain unchanged and the ignored annotations will
        be removed.

        Args:
            data_infos (list[dict]): Annotation infos.
            ann_shot_filter (dict): Specific which class and how many
                instances of each class to load from annotation file.
                For example: {'dog': 10, 'cat': 10, 'person': 5}

        Returns:
            list[dict]: Annotation infos where number of specified class
                shots less than or equal to predefined number.
        """
        if ann_shot_filter is None:
            return data_infos
        # build instance indices of (img_id, gt_idx)
        filter_instances = {key: [] for key in ann_shot_filter.keys()}
        keep_instances_indices = []
        for idx, data_info in enumerate(data_infos):
            ann = data_info
            for i in range(ann['labels'].shape[0]):
                instance_class_name = self.CLASSES[ann['labels'][i]]
                # only filter instance from the filter class
                if instance_class_name in ann_shot_filter.keys():
                    filter_instances[instance_class_name].append((idx, i))
                # skip the class not in the filter
                else:
                    keep_instances_indices.append((idx, i))
        # keep the selected annotations and remove the undesired annotations
        new_data_infos = []
        instance_class_name = list(filter_instances.keys())[0]
        for idx, data_info in enumerate(data_infos):
            selected_instance_indices = \
                sorted([instance[1] for instance in filter_instances[instance_class_name]
                        if instance[0] == idx])
            if len(selected_instance_indices) == 0:
                selected_ann = dict(
                    bboxes=np.empty((0, 4)),
                    labels=np.empty((0, )),
                )
                new_data_infos.append(selected_ann)
                continue
            # ann = data_info['ann']
            ann = data_info
            selected_ann = dict(
                bboxes=ann['bboxes'][selected_instance_indices],
                labels=ann['labels'][selected_instance_indices],
            )
            new_data_infos.append(selected_ann)
        return new_data_infos

    def get_classes(self, classes: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(classes, str):
            assert classes in self.SPLIT.keys(
            ), f'{self.dataset_name}: not a pre-defined classes or ' \
               f'split in VHR10_SPLIT'
            class_names = self.SPLIT[classes]
            if 'BASE_CLASSES' in classes:
                assert self.num_novel_shots is None, \
                    f'{self.dataset_name}: BASE_CLASSES do not have ' \
                    f'novel instances.'
            elif 'NOVEL_CLASSES' in classes:
                assert self.num_base_shots is None, \
                    f'{self.dataset_name}: NOVEL_CLASSES do not have ' \
                    f'base instances.'
                self.split_id = int(classes[-1])
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names

    def load_annotations(self, ann_file):
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        data_infos = []
        ann_file = ann_file[0]['ann_file']
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            file_name = osp.join(self.img_subdir, f'{img_id}.jpg')
            txt_path = osp.join(self.img_prefix, self.ann_subdir,
                                f'{img_id}.txt')
            img = Image.open(osp.join(self.img_prefix, file_name))
            width, height = img.size
            ann_info = self._parse_txt_ann_info(txt_path)
            data_infos.append(
                dict(
                    id=img_id,
                    filename=file_name,
                    width=width,
                    height=height,
                    ann=ann_info))

        return data_infos

    def evaluate(self,
                 results: List[Sequence],
                 metric: Union[str, List[str]] = 'mAP',
                 logger: Optional[object] = None,
                 proposal_nums: Sequence[int] = (100, 300, 1000, 2000, 10000),
                 iou_thr: Optional[Union[float, Sequence[float]]] = [
                     0.5,
                 ],
                 class_splits: Optional[List[str]] = None) -> Dict:
        for i in range(len(results)):
            for j in range(len(results[i])):
                for k in range(4):
                    results[i][j][:, k] -= self.coordinate_offset[k]

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        if class_splits is not None:
            for k in class_splits:
                assert k in self.SPLIT.keys(), 'undefiend classes split.'
            class_splits = {k: self.SPLIT[k] for k in class_splits}
            class_splits_mean_aps = {k: [] for k in class_splits.keys()}

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, ap_results = eval_map(
                    results,
                    annotations,
                    classes=self.CLASSES,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset='voc07',
                    logger=logger,
                    use_legacy_coordinate=True)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)

                # calculate evaluate results of different class splits
                if class_splits is not None:
                    for k in class_splits.keys():
                        aps = [
                            cls_results['ap']
                            for i, cls_results in enumerate(ap_results)
                            if self.CLASSES[i] in class_splits[k]
                        ]
                        class_splits_mean_ap = np.array(aps).mean().item()
                        class_splits_mean_aps[k].append(class_splits_mean_ap)
                        eval_results[
                            f'{k}: AP{int(iou_thr * 100):02d}'] = round(
                                class_splits_mean_ap, 3)

            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            if class_splits is not None:
                for k in class_splits.keys():
                    mAP = sum(class_splits_mean_aps[k]) / len(
                        class_splits_mean_aps[k])
                    print_log(f'{k} mAP: {mAP}', logger=logger)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            new_results = []
            for reslut_per_img in results:
                temp = np.empty(shape=(0, 5))
                for i in range(len(reslut_per_img)):
                    if reslut_per_img[i].size != 0:
                        temp = np.concatenate((temp, reslut_per_img[i]),
                                              axis=0)
                new_results.append(temp)

            recalls = eval_recalls(
                gt_bboxes, new_results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]

            # calculate per class AR
            per_class_recall = []
            for cls in self.CLASSES:
                filter_result = self._select_annotations(
                    annotations, {cls: 1000})
                gt_bboxes = [ann['bboxes'] for ann in filter_result]
                recalls = eval_recalls(
                    gt_bboxes,
                    new_results,
                    proposal_nums,
                    iou_thr,
                    logger=logger,
                    silent=True)
                per_class_recall.append(recalls)
            print_per_class_recall(
                per_class_recall,
                self.CLASSES,
                iou_thr,
                proposal_idx=2,
                logger=logger)
        return eval_results


@DATASETS.register_module()
class FewShotVHR10DatasetV2(FewShotVHR10Dataset):

    def _parse_txt_ann_info(self, txt_path):
        with open(txt_path) as f:
            annotations = f.readlines()
            f.close()

        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        # parase annotations
        for annotation in annotations:
            if annotation == '\n':  # skip file ends
                continue
            ignore = False
            # label transform
            label = int(annotation.split(',')[-1]) - 1
            classes_name = self.SPLIT['CLASSES'][label]
            if classes_name not in self.CLASSES:
                return None
            else:
                label = self.cat2label[classes_name]

            coordinates = re.findall(r'[(](.*?)[)]', annotation)
            x1, y1 = coordinates[0].split(',')
            x2, y2 = coordinates[1].split(',')
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            if not self.test_mode:
                bbox = [
                    i + offset
                    for i, offset in zip(bbox, self.coordinate_offset)
                ]

            # ignore too small objs in training
            if self.min_size is not None:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)

        #
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))

        return ann

    def load_annotations(self, ann_file):
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        data_infos = []
        ann_file = ann_file[0]['ann_file']
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            file_name = osp.join(self.img_subdir, f'{img_id}.jpg')
            txt_path = osp.join(self.img_prefix, self.ann_subdir,
                                f'{img_id}.txt')
            img = Image.open(osp.join(self.img_prefix, file_name))
            width, height = img.size
            ann_info = self._parse_txt_ann_info(txt_path)
            if ann_info is not None:
                data_infos.append(
                    dict(
                        id=img_id,
                        filename=file_name,
                        width=width,
                        height=height,
                        ann=ann_info))
            else:
                continue
        return data_infos


@DATASETS.register_module()
class FewShotVHR10DefaultDataset(FewShotVHR10Dataset):
    vhr10_benchmark = {
        f'{cls}_SPLIT{split}_{shot}SHOT': [
            dict(
                type='ann_file',
                ann_file=f'data/few_shot_ann/vhr10/benchmark_{shot}shot/'
                f'box_{shot}shot_{class_name}_train.txt',
                ann_classes=[class_name])
            for class_name in VHR10_SPLIT[f'{cls}_SPLIT{split}']
        ]
        for cls in ['ALL_CLASSES', 'BASE_CLASSES', 'NOVEL_CLASSES']
        for shot in [3, 5, 10, 20] for split in [1, 2]
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def load_annotations(self, ann_folder):
        """
            Args:
                ann_folder: folder that contains vhr10  annotations xml files
        """
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        img_ids = []
        setting = self.ann_cfg[0]['setting']
        split = self.ann_cfg[0]['split']
        classes = self.classes.split('_SPLIT')[0]
        benchmarks = self.vhr10_benchmark[classes + '_' + split + '_' +
                                          setting]
        for benchmark in benchmarks:
            benchmark_file = benchmark['ann_file']
            with open(benchmark_file) as f:
                s = f.readlines()
                for si in s:
                    img_ids.append(si.rstrip('\n'))

        data_infos = []
        for img_id in img_ids:
            file_name = osp.join(self.img_subdir, f'{img_id}.jpg')
            txt_path = osp.join(self.img_prefix, self.ann_subdir,
                                f'{img_id}.txt')
            img = Image.open(osp.join(self.img_prefix, file_name))
            width, height = img.size
            ann_info = self._parse_txt_ann_info(txt_path)
            data_infos.append(
                dict(
                    id=img_id,
                    filename=file_name,
                    width=width,
                    height=height,
                    ann=ann_info))

        return data_infos


@DATASETS.register_module()
class FewShotVHR10CopyDataset(FewShotVHR10Dataset):

    def __init__(self, ann_cfg: Union[List[Dict], Dict], **kwargs) -> None:
        super().__init__(ann_cfg=ann_cfg, **kwargs)

    def ann_cfg_parser(self, ann_cfg: Union[List[Dict], Dict]) -> List[Dict]:
        """Parse annotation config from a copy of other dataset's `data_infos`.

        Args:
            ann_cfg (list[dict] | dict): contain `data_infos` from other
                dataset. Example:
                [dict(data_infos=FewShotVOCDataset.data_infos)]

        Returns:
            list[dict]: Annotation information.
        """
        data_infos = []
        if isinstance(ann_cfg, dict):
            assert ann_cfg.get('data_infos', None) is not None, \
                f'{self.dataset_name}: ann_cfg of ' \
                f'FewShotVHR10CopyDataset require data_infos.'
            # directly copy data_info
            data_infos = ann_cfg['data_infos']
        elif isinstance(ann_cfg, list):
            for ann_cfg_ in ann_cfg:
                assert ann_cfg_.get('data_infos', None) is not None, \
                    f'{self.dataset_name}: ann_cfg of ' \
                    f'FewShotVHR10CopyDataset require data_infos.'
                # directly copy data_info
                data_infos += ann_cfg_['data_infos']
        return data_infos
