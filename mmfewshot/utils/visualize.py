# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from mmdet.core.visualization.image import draw_bboxes, draw_labels
# from mmrotate.core.visualization.image import draw_rbboxes
# from matplotlib.patches import Circle
# from matplotlib.collections import PatchCollection

# def draw_points(ax, points, color='g', alpha=0.8):
#     """Draw bounding boxes on the axes.

#     Args:
#         ax (matplotlib.Axes): The input axes.
#         points (ndarray): The input bounding boxes with the shape
#             of (n, 2).
#         color (list[tuple] | matplotlib.color): the colors for each
#             bounding boxes.
#     Returns:
#         matplotlib.Axes: The result axes.
#     """
#     Circles = []
#     for i, point in enumerate(points):
#         np_point = np.array(point).reshape(2)
#         Circles.append(Circle(np_point))
#     p = PatchCollection(
#         Circles,
#         edgecolors=color,
#         alpha=alpha)
#     ax.add_collection(p)

#     return ax

# def visualize_torch_tensor(input_tensor: torch.Tensor):
#     """visualize a torch tensor

#     Args:
#         input_tensor (torch.Tensor): _description_
#     """
#     image = input_tensor.clone()
#     assert image.ndim == 3, 'the shape of input tensor should be equal to 3'
#     # change shape (N,H,W) to (H,W,N)
#     if image.shape[0] == 1 or image.shape[0] == 3:
#         image = image.permute(1, 2, 0)
#     image = image.detach().cpu().numpy()
#     plt.imshow(image)
#     plt.show()

# def visualize_boxes(img_meta, bboxes, labels=None):
#     """visualize image with bboxes

#     Args:
#         img_meta (str): image_meta
#         bboxes (np.array): shape of (n,4) xyxy format
#             or shape of (n, 5) xywha
#     """
#     # data preprocess
#     if isinstance(bboxes, torch.Tensor):
#         bboxes = bboxes.clone().detach().cpu().numpy()
#     if isinstance(labels, torch.Tensor):
#         labels = labels.clone().detach().cpu().numpy()

#     # load img_meta params
#     image_path = img_meta['filename']
#     image_flip = img_meta.get('flip')
#     if image_flip:
#         flip_direction = img_meta['flip_direction']

#     # flip image
#     image = cv2.imread(image_path)
#     if image_flip:
#         if flip_direction == 'horizontal':
#             image = np.fliplr(image)
#         elif flip_direction == 'vertical':
#             image = np.flipud(image)

#     # @TODO: pad image and resize image

#     # show image
#     plt.imshow(image)
#     # shot horizontal bboxes
#     currentAxis = plt.gca()
#     if bboxes.shape[-1] == 5:
#         currentAxis = draw_rbboxes(currentAxis, bboxes, color='r')
#     else:
#         currentAxis = draw_bboxes(currentAxis, bboxes, color='r')
#     # show labels
#     if labels is not None:
#         draw_labels(currentAxis, labels, positions=bboxes[:, :2])
#     plt.show()

# def visualize_points(img_meta, points, labels=None):
#     # data preprocess
#     if isinstance(points, torch.Tensor):
#         points = points.clone().detach().cpu().numpy()
#     if isinstance(labels, torch.Tensor):
#         labels = labels.clone().detach().cpu().numpy()

#     # load img_meta params
#     image_path = img_meta['filename']
#     image_flip = img_meta.get('flip')
#     if image_flip:
#         flip_direction = img_meta['flip_direction']

#     # flip image
#     image = cv2.imread(image_path)
#     if image_flip:
#         if flip_direction == 'horizontal':
#             image = np.fliplr(image)
#         elif flip_direction == 'vertical':
#             image = np.flipud(image)

#     # show image
#     plt.imshow(image)
#     # shot horizontal bboxes
#     currentAxis = plt.gca()
#     currentAxis = draw_points(currentAxis, points, color='r')
#     if labels is not None:
#         draw_labels(currentAxis, labels, positions=points)
#     plt.show()
