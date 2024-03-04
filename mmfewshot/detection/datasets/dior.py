import os.path as osp
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.xml_style import XMLDataset
from PIL import Image

from mmfewshot.detection.core import eval_map
from mmfewshot.detection.datasets.base import BaseFewShotDataset
from mmfewshot.utils import get_root_logger
from ..evaluation.recall import eval_recalls, print_per_class_recall

DIOR_SPLIT = dict(
    ALL_CLASSES_SPLIT1=('airplane', 'airport', 'dam',
                        'Expressway-Service-area', 'Expressway-toll-station',
                        'golffield', 'groundtrackfield', 'harbor', 'overpass',
                        'stadium', 'storagetank', 'tenniscourt',
                        'trainstation', 'vehicle', 'windmill', 'baseballfield',
                        'basketballcourt', 'bridge', 'chimney', 'ship'),
    ALL_CLASSES_SPLIT2=(
        'basketballcourt',
        'bridge',
        'chimney',
        'dam',
        'Expressway-toll-station',
        'golffield',
        'overpass',
        'ship',
        'stadium',
        'storagetank',
        'vehicle',
        'baseballfield',
        'tenniscourt',
        'trainstation',
        'windmill',
        'airplane',
        'airport',
        'Expressway-Service-area',
        'harbor',
        'groundtrackfield',
    ),
    ALL_CLASSES_SPLIT3=('airplane', 'airport', 'baseballfield',
                        'basketballcourt', 'bridge', 'chimney',
                        'Expressway-Service-area', 'Expressway-toll-station',
                        'groundtrackfield', 'harbor', 'overpass', 'ship',
                        'stadium', 'trainstation', 'windmill', 'dam',
                        'golffield', 'storagetank', 'tenniscourt', 'vehicle'),
    ALL_CLASSES_SPLIT4=('airport', 'basketballcourt', 'bridge', 'chimney',
                        'dam', 'Expressway-toll-station', 'golffield',
                        'groundtrackfield', 'harbor', 'ship', 'storagetank',
                        'airplane', 'baseballfield', 'tenniscourt', 'windmill',
                        'Expressway-Service-area', 'overpass', 'stadium',
                        'trainstation', 'vehicle'),
    ALL_CLASSES_SPLIT5=('airport', 'basketballcourt', 'bridge', 'chimney',
                        'dam', 'Expressway-Service-area',
                        'Expressway-toll-station', 'golffield',
                        'groundtrackfield', 'harbor', 'overpass', 'ship',
                        'stadium', 'storagetank', 'vehicle', 'airplane',
                        'baseballfield', 'tenniscourt', 'trainstation',
                        'windmill'),
    NOVEL_CLASSES_SPLIT1=('baseballfield', 'basketballcourt', 'bridge',
                          'chimney', 'ship'),
    NOVEL_CLASSES_SPLIT2=(
        'airplane',
        'airport',
        'Expressway-Service-area',
        'harbor',
        'groundtrackfield',
    ),
    NOVEL_CLASSES_SPLIT3=('dam', 'golffield', 'storagetank', 'tenniscourt',
                          'vehicle'),
    NOVEL_CLASSES_SPLIT4=('Expressway-Service-area', 'overpass', 'stadium',
                          'trainstation', 'windmill'),
    NOVEL_CLASSES_SPLIT5=('airplane', 'baseballfield', 'tenniscourt',
                          'trainstation', 'windmill'),
    BASE_CLASSES_SPLIT1=('airplane', 'airport', 'dam',
                         'Expressway-Service-area', 'Expressway-toll-station',
                         'golffield', 'groundtrackfield', 'harbor', 'overpass',
                         'stadium', 'storagetank', 'tenniscourt',
                         'trainstation', 'vehicle', 'windmill'),
    BASE_CLASSES_SPLIT2=('basketballcourt', 'bridge', 'chimney', 'dam',
                         'Expressway-toll-station', 'golffield', 'overpass',
                         'ship', 'stadium', 'storagetank', 'vehicle',
                         'baseballfield', 'tenniscourt', 'trainstation',
                         'windmill'),
    BASE_CLASSES_SPLIT3=('airplane', 'airport', 'baseballfield',
                         'basketballcourt', 'bridge', 'chimney',
                         'Expressway-Service-area', 'Expressway-toll-station',
                         'groundtrackfield', 'harbor', 'overpass', 'ship',
                         'stadium', 'trainstation', 'windmill'),
    BASE_CLASSES_SPLIT4=('airport', 'basketballcourt', 'bridge', 'chimney',
                         'dam', 'Expressway-toll-station', 'golffield',
                         'groundtrackfield', 'harbor', 'ship', 'storagetank',
                         'airplane', 'baseballfield', 'tenniscourt',
                         'vehicle'),
    BASE_CLASSES_SPLIT5=('airport', 'basketballcourt', 'bridge', 'chimney',
                         'dam', 'Expressway-Service-area',
                         'Expressway-toll-station', 'golffield',
                         'groundtrackfield', 'harbor', 'overpass', 'ship',
                         'stadium', 'storagetank', 'vehicle'))


@DATASETS.register_module()
class FewShotDIORDatasetSkipNovel(BaseFewShotDataset):

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
                 **kwargs) -> None:
        if dataset_name is None:
            self.dataset_name = 'Test dataset' \
                if test_mode else 'Train dataset'
        else:
            self.dataset_name = dataset_name

        self.SPLIT = DIOR_SPLIT
        self.split_id = None

        assert classes is not None, f'{self.dataset_name}: classes in ' \
                                    f'`FewShotVOCDataset` can not be None.'
        self.num_novel_shots = num_novel_shots
        self.num_base_shots = num_base_shots
        self.min_bbox_area = min_bbox_area
        self.classes = classes
        self.CLASSES = self.get_classes(classes)

        if ann_shot_filter is None:
            # configure ann_shot_filter by num_novel_shots and num_base_shots
            if num_novel_shots is not None or num_base_shots is not None:
                ann_shot_filter = self._create_ann_shot_filter()
        else:
            assert num_novel_shots is None and num_base_shots is None, \
                f'{self.dataset_name}: can not config ann_shot_filter and ' \
                f'num_novel_shots/num_base_shots at the same time.'
        self.coordinate_offset = coordinate_offset
        self.use_difficult = use_difficult
        super().__init__(
            classes=None,
            ann_shot_filter=ann_shot_filter,
            dataset_name=dataset_name,
            test_mode=test_mode,
            **kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        data_infos = []
        ann_file = ann_file[0]['ann_file']
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = osp.join('JPEGImages', f'{img_id}.jpg')
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.img_prefix, 'JPEGImages',
                                    f'{img_id}.jpg')
                img = Image.open(img_path)
                width, height = img.size
            ann_info = self._get_xml_ann_info(img_id)
            if ann_info is None:
                continue
            data_infos.append(
                dict(
                    id=img_id,
                    filename=filename,
                    width=width,
                    height=height,
                    ann=ann_info))

        return data_infos

    def _get_xml_ann_info(self,
                          img_id: str,
                          classes: Optional[List[str]] = None) -> Dict:
        """get xml annotation info.

        Args:
            img_id (str): image id
            classes (Optional[List[str]], optional): classes to load. Defaults to None.

        Returns:
            Dict: Dict of an info or None
        """
        if classes is None:
            classes = self.CLASSES
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        xml_path = osp.join(self.img_prefix, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            # if an image contain annotations belongs to other classes may
            # decrease accuracy of new classes
            if name not in classes:
                return None
            label = self.cat2label[name]
            if self.use_difficult:
                difficult = 0
            else:
                difficult = obj.find('difficult')
                difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find('bndbox')
            # will add 1 for inverse of data loading logic.
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            if not self.test_mode:
                bbox = [
                    i + offset
                    for i, offset in zip(bbox, self.coordinate_offset)
                ]
            ignore = False
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
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
        ann_info = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann_info

    def get_classes(self, classes: Union[str, Sequence[str]]) -> List[str]:
        """Get class names.

        It supports to load pre-defined classes splits.
        The pre-defined classes splits are:
        ['ALL_CLASSES_SPLIT1', 'ALL_CLASSES_SPLIT2', 'ALL_CLASSES_SPLIT3',
         'BASE_CLASSES_SPLIT1', 'BASE_CLASSES_SPLIT2', 'BASE_CLASSES_SPLIT3',
         'NOVEL_CLASSES_SPLIT1','NOVEL_CLASSES_SPLIT2','NOVEL_CLASSES_SPLIT3']

        Args:
            classes (str | Sequence[str]): Classes for model training and
                provide fixed label for each class. When classes is string,
                it will load pre-defined classes in `FewShotVOCDataset`.
                For example: 'NOVEL_CLASSES_SPLIT1'.

        Returns:
            list[str]: List of class names.
        """
        # configure few shot classes setting
        if isinstance(classes, str):
            assert classes in self.SPLIT.keys(
            ), f'{self.dataset_name}: not a pre-defined classes or ' \
               f'split in DIOR_SPLIT'
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

    def _create_ann_shot_filter(self) -> Dict[str, int]:
        """Generate `ann_shot_filter` for novel and base classes.

        Returns:
            dict[str, int]: The number of shots to keep for each class.
        """
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
                     min_size: int = 32,
                     min_bbox_area: Optional[int] = None) -> List[int]:
        """Filter images not meet the demand.

        Args:
            min_size (int): Filter images with length or width
                smaller than `min_size`. Default: 32.
            min_bbox_area (int | None): Filter images with bbox whose
                area smaller `min_bbox_area`. If set to None, skip
                this filter. Default: None.

        Returns:
            list[int]: valid indices of `data_infos`.
        """
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

    # def evaluate(self,
    #              results: List[Sequence],
    #              metric: Union[str, List[str]] = 'mAP',
    #              logger: Optional[object] = None,
    #              proposal_nums: Sequence[int] = (100, 300, 1000, 2000),
    #              iou_thr: Optional[Union[float, Sequence[float]]] = [
    #                  0.5,
    #              ],
    #              class_splits: Optional[List[str]] = None) -> Dict:
    #     for i in range(len(results)):
    #         for j in range(len(results[i])):
    #             for k in range(4):
    #                 results[i][j][:, k] -= self.coordinate_offset[k]

    #     if not isinstance(metric, str):
    #         assert len(metric) == 1
    #         metric = metric[0]
    #     allowed_metrics = ['mAP', 'recall']
    #     if metric not in allowed_metrics:
    #         raise KeyError(f'metric {metric} is not supported')
    #     if class_splits is not None:
    #         for k in class_splits:
    #             assert k in self.SPLIT.keys(), 'undefiend classes split.'
    #         class_splits = {k: self.SPLIT[k] for k in class_splits}
    #         class_splits_mean_aps = {k: [] for k in class_splits.keys()}

    #     annotations = [self.get_ann_info(i) for i in range(len(self))]
    #     eval_results = OrderedDict()
    #     iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
    #     if metric == 'mAP':
    #         assert isinstance(iou_thrs, list)
    #         mean_aps = []
    #         for iou_thr in iou_thrs:
    #             print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
    #             mean_ap, ap_results = eval_map(
    #                 results,
    #                 annotations,
    #                 classes=self.CLASSES,
    #                 scale_ranges=None,
    #                 iou_thr=iou_thr,
    #                 dataset='voc07',
    #                 logger=logger,
    #                 use_legacy_coordinate=True)
    #             mean_aps.append(mean_ap)
    #             eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)

    #             # calculate evaluate results of different class splits
    #             if class_splits is not None:
    #                 for k in class_splits.keys():
    #                     aps = [
    #                         cls_results['ap']
    #                         for i, cls_results in enumerate(ap_results)
    #                         if self.CLASSES[i] in class_splits[k]
    #                     ]
    #                     class_splits_mean_ap = np.array(aps).mean().item()
    #                     class_splits_mean_aps[k].append(class_splits_mean_ap)
    #                     eval_results[
    #                         f'{k}: AP{int(iou_thr * 100):02d}'] = round(
    #                             class_splits_mean_ap, 3)

    #         eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
    #         if class_splits is not None:
    #             for k in class_splits.keys():
    #                 mAP = sum(class_splits_mean_aps[k]) / len(
    #                     class_splits_mean_aps[k])
    #                 print_log(f'{k} mAP: {mAP}', logger=logger)
    #     elif metric == 'recall':
    #         gt_bboxes = [ann['bboxes'] for ann in annotations]
    #         new_results = []
    #         for reslut_per_img in results:
    #             temp = np.empty(shape=(0, 5))
    #             for i in range(len(reslut_per_img)):
    #                 if reslut_per_img[i].size != 0:
    #                     temp = np.concatenate((temp, reslut_per_img[i]),
    #                                           axis=0)
    #             new_results.append(temp)

    #         recalls = eval_recalls(
    #             gt_bboxes, new_results, proposal_nums, iou_thr, logger=logger)
    #         for i, num in enumerate(proposal_nums):
    #             for j, iou in enumerate(iou_thrs):
    #                 eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
    #         if recalls.shape[1] > 1:
    #             ar = recalls.mean(axis=1)
    #             for i, num in enumerate(proposal_nums):
    #                 eval_results[f'AR@{num}'] = ar[i]

    #         # calculate per class AR
    #         per_class_recall = []
    #         for cls in self.CLASSES:
    #             filter_result = self._select_annotations(
    #                 annotations, {cls: 1000})
    #             gt_bboxes = [ann['bboxes'] for ann in filter_result]
    #             recalls = eval_recalls(
    #                 gt_bboxes,
    #                 new_results,
    #                 proposal_nums,
    #                 iou_thr,
    #                 logger=logger,
    #                 silent=True)
    #             per_class_recall.append(recalls)
    #         print_per_class_recall(
    #             per_class_recall,
    #             self.CLASSES,
    #             iou_thr,
    #             proposal_idx=2,
    #             logger=logger)
    #     return eval_results

    def evaluate(self,
                 results: List[Sequence],
                 metric: Union[str, List[str]] = 'mAP',
                 logger: Optional[object] = None,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thr: Optional[Union[float, Sequence[float]]] = 0.5,
                 class_splits: Optional[List[str]] = None) -> Dict:
        """Evaluation in VOC protocol and summary results of different splits
        of classes.

        Args:
            results (list[list | tuple]): Predictions of the model.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'. Default: mAP.
            logger (logging.Logger | None): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            class_splits: (list[str] | None): Calculate metric of classes
                split  defined in VOC_SPLIT. For example:
                ['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'].
                Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """
        # It should be noted that in the original mmdet implementation,
        # the four coordinates are reduced by 1 when the annotation
        # is parsed. Here we following detectron2, only xmin and ymin
        # will be reduced by 1 during training. The groundtruth used for
        # evaluation or testing keep consistent with original xml
        # annotation file and the xmin and ymin of prediction results
        # will add 1 for inverse of data loading logic.
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
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results

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


@DATASETS.register_module()
class FewShotDIORDatasetV2(FewShotDIORDatasetSkipNovel):

    def _get_xml_ann_info(self,
                          img_id: str,
                          classes: Optional[List[str]] = None) -> Dict:
        """get xml annotation info.

        Args:
            img_id (str): image id
            classes (Optional[List[str]], optional): classes to load. Defaults to None.

        Returns:
            Dict: Dict of an info or None
        """
        if classes is None:
            classes = self.CLASSES
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        xml_path = osp.join(self.img_prefix, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if self.use_difficult:
                difficult = 0
            else:
                difficult = obj.find('difficult')
                difficult = 0 if difficult is None else int(difficult.text)
            if name not in classes:
                difficult = 1
                label = len(classes)
            else:
                label = self.cat2label[name]
            bnd_box = obj.find('bndbox')
            # will add 1 for inverse of data loading logic.
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            if not self.test_mode:
                bbox = [
                    i + offset
                    for i, offset in zip(bbox, self.coordinate_offset)
                ]
            ignore = False
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
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
        ann_info = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann_info


@DATASETS.register_module()
class FewShotDIORDataset(FewShotDIORDatasetV2):

    def _get_xml_ann_info(self,
                          img_id: str,
                          classes: Optional[List[str]] = None) -> Dict:
        """get xml annotation info.

        Args:
            img_id (str): image id
            classes (Optional[List[str]], optional): classes to load. Defaults to None.

        Returns:
            Dict: Dict of an info or None
        """
        if classes is None:
            classes = self.CLASSES
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        xml_path = osp.join(self.img_prefix, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if self.use_difficult:
                difficult = 0
            else:
                difficult = obj.find('difficult')
                difficult = 0 if difficult is None else int(difficult.text)
            if name not in classes:
                continue
            label = self.cat2label[name]
            bnd_box = obj.find('bndbox')
            # will add 1 for inverse of data loading logic.
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            if not self.test_mode:
                bbox = [
                    i + offset
                    for i, offset in zip(bbox, self.coordinate_offset)
                ]
            ignore = False
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
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
        ann_info = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann_info


@DATASETS.register_module()
class FewShotDIORDefaultDataset(FewShotDIORDataset):
    """Dataset with some pre-defined DIOR annotation paths.

    :obj:`FewShotDIORDefaultDataset` provides pre-defined annotation files
    to ensure the reproducibility. The pre-defined annotation files provide
    fixed training data to avoid random sampling. The usage of `ann_cfg' is
    different from :obj:`FewShotDIORDataset`. The `ann_cfg' should contain
    two filed: `method` and `setting`.

    Args:
        ann_cfg (list[dict]): Each dict should contain
            `method` and `setting` to get corresponding
            annotation from `DEFAULT_ANN_CONFIG`.
            For example: [dict(method='dota', setting='SPILT1_1shot')].
    """
    dior_benchmark = {
        f'{cls}_SPLIT{split}_{shot}SHOT': [
            dict(
                type='ann_file',
                ann_file=f'data/few_shot_ann/dior/benchmark_{shot}shot/'
                f'box_{shot}shot_{class_name}_train.txt',
                ann_classes=[class_name])
            for class_name in DIOR_SPLIT[f'{cls}_SPLIT{split}']
        ]
        for cls in ['ALL_CLASSES', 'BASE_CLASSES', 'NOVEL_CLASSES']
        for shot in [1, 2, 3, 5, 10, 20] for split in [1, 2, 3, 4, 5]
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def load_annotations(self, ann_folder):
        """
            Args:
                ann_folder: folder that contains DIOR v1 annotations xml files
        """
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        img_ids = []
        setting = self.ann_cfg[0]['setting']
        split = self.ann_cfg[0]['split']
        classes = self.classes.split('_SPLIT')[0]
        benchmarks = self.dior_benchmark[classes + '_' + split + '_' + setting]
        for benchmark in benchmarks:
            benchmark_file = benchmark['ann_file']
            with open(benchmark_file) as f:
                s = f.readlines()
                for si in s:
                    img_ids.append(si.rstrip('\n'))

        data_infos = []
        for img_id in img_ids:
            filename = osp.join('JPEGImages', f'{img_id}.jpg')
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.img_prefix, 'JPEGImages',
                                    f'{img_id}.jpg')
                img = Image.open(img_path)
                width, height = img.size
            ann_info = self._get_xml_ann_info(img_id)
            data_infos.append(
                dict(
                    id=img_id,
                    filename=filename,
                    width=width,
                    height=height,
                    ann=ann_info))

        return data_infos


@DATASETS.register_module()
class FewShotDIORDefaultDatasetWithSeedK(FewShotDIORDataset):

    def __init__(self, seed=None, **kwargs) -> None:
        self.seed = seed
        super().__init__(**kwargs)

    dior_benchmark = {
        f'{cls}_SPLIT{split}_{shot}SHOT_SEED{seed}': [
            dict(
                type='ann_file',
                ann_file=f'data/few_shot_ann/dior_cir_fsd/seed{seed}/'
                f'box_{shot}shot_{class_name}_train.txt',
                ann_classes=[class_name])
            for class_name in DIOR_SPLIT[f'{cls}_SPLIT{split}']
        ]
        for cls in ['ALL_CLASSES', 'BASE_CLASSES', 'NOVEL_CLASSES']
        for shot in [1, 2, 3, 5, 10, 20] for split in [1, 2, 3, 4, 5]
        for seed in range(1, 22)
    }

    def load_annotations(self, ann_folder):
        """
            Args:
                ann_folder: folder that contains DIOR v1 annotations xml files
        """
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        img_ids = []
        setting = self.ann_cfg[0]['setting']
        split = self.ann_cfg[0]['split']
        seed = self.seed
        classes = self.classes.split('_SPLIT')[0]
        benchmarks = self.dior_benchmark[classes + '_' + split + '_' +
                                         setting + '_' + 'SEED' + str(seed)]
        for benchmark in benchmarks:
            benchmark_file = benchmark['ann_file']
            with open(benchmark_file) as f:
                s = f.readlines()
                for si in s:
                    img_ids.append(si.rstrip('\n'))

        data_infos = []
        for img_id in img_ids:
            filename = osp.join('JPEGImages', f'{img_id}.jpg')
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.img_prefix, 'JPEGImages',
                                    f'{img_id}.jpg')
                img = Image.open(img_path)
                width, height = img.size
            ann_info = self._get_xml_ann_info(img_id)
            data_infos.append(
                dict(
                    id=img_id,
                    filename=filename,
                    width=width,
                    height=height,
                    ann=ann_info))

        return data_infos


@DATASETS.register_module()
class FewShotDIORCopyDataset(FewShotDIORDataset):
    """Copy other DIOR few shot datasets' `data_infos` directly.

    This dataset is mainly used for model initialization in some meta-learning
    detectors. In their cases, the support data are randomly sampled
    during training phase and they also need to be used in model
    initialization before evaluation. To copy the random sampling results,
    this dataset supports to load `data_infos` of other datasets via `ann_cfg`

    Args:
        ann_cfg (list[dict] | dict): contain `data_infos` from other
            dataset. Example: [dict(data_infos=FewShotVOCDataset.data_infos)]
    """

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
                f'FewShotDIORCopyDataset require data_infos.'
            # directly copy data_info
            data_infos = ann_cfg['data_infos']
        elif isinstance(ann_cfg, list):
            for ann_cfg_ in ann_cfg:
                assert ann_cfg_.get('data_infos', None) is not None, \
                    f'{self.dataset_name}: ann_cfg of ' \
                    f'FewShotDIORCopyDataset require data_infos.'
                # directly copy data_info
                data_infos += ann_cfg_['data_infos']
        return data_infos
