import cv2
import numpy as np
import torch
from copy import deepcopy


def plot_keypoints(image: torch.tensor, kpts: torch.tensor,
                   radius: int = 2, color=(255, 0, 0)) -> np.ndarray:
    """ visualize keypoints on image
    :param image: [H, W, 3] in range [0, 1]
    :param kpts:  [N, 2] in range [0, 1]
    :param radius: radius of the keypoint
    :param color:  color of the keypoint
    :return: image with keypoints
    """
    image = image.cpu().detach().numpy() if isinstance(image, torch.Tensor) else image
    kpts = kpts.cpu().detach().numpy() if isinstance(kpts, torch.Tensor) else kpts

    if image.dtype is not np.dtype('uint8'):
        image = image * 255
        image = image.astype(np.uint8)

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    H, W, _ = image.shape
    out = np.ascontiguousarray(deepcopy(image))
    pts = kpts[:, 0:2] * np.array([W, H])
    pts = np.round(pts).astype(int)

    for pt in pts:
        x0, y0 = pt
        cv2.drawMarker(out, (x0, y0), color, cv2.MARKER_CROSS, radius)
    return out


def plot_dense_matches(img0: torch.tensor, img1: torch.tensor,
                       matches: torch.tensor) -> np.ndarray:
    """ visualize dense matches
    :param img0: [H, W, 3] in range [0, 1]
    :param img1: [H, W, 3] in range [0, 1]
    :param matches: [2, H/64, W/64] in range [0, H/64 * W/64)
    """
    matches01 = matches[0]
    matches10 = matches[1]
    show = np.concatenate([img0.numpy(), img1.numpy()], axis=1)
    show = (show * 255).astype(np.uint8)  # convert to uint8
    if len(show.shape) == 2 or show.shape[2] == 1:
        show = cv2.cvtColor(show, cv2.COLOR_GRAY2RGB)
    for i in range(matches01.shape[0] * matches01.shape[1]):
        id_x = i % matches01.shape[1]
        id_y = i // matches01.shape[1]
        if matches01[id_y, id_x] != -1:
            matched_id = matches01[id_y, id_x]
            cv2.line(show, (id_x * 64 + 32, id_y * 64 + 32),
                     (int(matched_id) % matches01.shape[1] * 64 + img0.shape[1] + 32,
                      int(matched_id) // matches01.shape[1] * 64 + 32),
                     (0, 255, 0), 1)

    for i in range(matches10.shape[0] * matches10.shape[1]):
        id_x = i % matches10.shape[1]
        id_y = i // matches10.shape[1]
        if matches10[id_y, id_x] != -1:
            matched_id = matches10[id_y, id_x]
            cv2.line(show, (id_x * 64 + img0.shape[1] + 32, id_y * 64 + 32),
                     (int(matched_id) % matches10.shape[1] * 64 + 32,
                      int(matched_id) // matches10.shape[1] * 64 + 32),
                     (0, 255, 0), 1)
    return show


def plot_matches(img0: torch.tensor, img1: torch.tensor,
                 pt1: torch.tensor, pt2: torch.tensor,
                 matches: torch.tensor, color=(0, 255, 0), thickness: int = 1) -> np.ndarray:
    """ visualize matches
    :param img0: [H, W, 3] in range [0, 1]
    :param img1: [H, W, 3] in range [0, 1]
    :param pt1: [N, 2] in range [0, 1]
    :param pt2: [M, 2] in range [0, 1]
    :param matches: [K, 2] in range [N, M]
    :param color: color of the line
    :param thickness: thickness of the line
    """
    pt1 = pt1.cpu().detach().numpy() if isinstance(pt1, torch.Tensor) else pt1
    pt2 = pt2.cpu().detach().numpy() if isinstance(pt2, torch.Tensor) else pt2
    pt1 = pt1 * np.array([img0.shape[1], img0.shape[0]])
    pt2 = pt2 * np.array([img1.shape[1], img1.shape[0]])
    pt1 = np.round(pt1).astype(int)
    pt2 = np.round(pt2).astype(int)
    pt2 = pt2 + np.array([img0.shape[1], 0])
    show = np.concatenate([img0.numpy(), img1.numpy()], axis=1)
    show = (show * 255).astype(np.uint8)  # convert to uint8
    if len(show.shape) == 2 or show.shape[2] == 1:
        show = cv2.cvtColor(show, cv2.COLOR_GRAY2RGB)
    for pt in pt1:
        x0, y0 = pt
        cv2.drawMarker(show, (x0, y0), color, cv2.MARKER_CROSS, 4)
    for pt in pt2:
        x0, y0 = pt
        cv2.drawMarker(show, (x0, y0), color, cv2.MARKER_CROSS, 4)
    for i in range(len(matches)):
        x0, y0 = pt1[matches[i][0]]
        x1, y1 = pt2[matches[i][1]]
        cv2.line(show, (x0, y0), (x1, y1), color, thickness)
    return show


def plot_op_matches(img0: torch.tensor, img1: torch.tensor,
                    pt1: torch.tensor, pt2: torch.tensor,
                    color=(0, 255, 0), thickness: int = 1) -> np.ndarray:
    """ visualize matches
    :param img0: [H, W, 3] in range [0, 1]
    :param img1: [H, W, 3] in range [0, 1]
    :param pt1: [N, 2] in range [0, 1]
    :param pt2: [N, 2] in range [0, 1]
    :param color: color of the line
    :param thickness: thickness of the line
    """
    show = np.concatenate([img0.numpy(), img1.numpy()], axis=1)
    show = (show * 255).astype(np.uint8)  # convert to uint8
    pt2 = pt2 + np.array([img0.shape[1], 0])
    for i in range(len(pt1)):
        cv2.line(show, (int(pt1[i][0]), int(pt1[i][1])), (int(pt2[i][0]), int(pt2[i][1])), color, thickness)
    return show


def plot_gt_matches(img0, img1, dense_matches):
    show = np.concatenate([img0, img1], axis=1)
    show = (show * 255).astype(np.uint8)  # convert to uint8
    if len(show.shape) == 2 or show.shape[2] == 1:
        show = cv2.cvtColor(show, cv2.COLOR_GRAY2RGB)
    h, w, _ = img0.shape
    for i in range(dense_matches.shape[0]):
        m = dense_matches[i]


if __name__ == "__main__":
    ''' test plot functions '''
    print("test plot functions")

