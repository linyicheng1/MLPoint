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


def plot_dense_matches(img0, img1,
                       xys, matches_gt) -> np.ndarray:
    """ visualize dense matches
    :param img0: [H, W, 3] in range [0, 1]
    :param img1: [H, W, 3] in range [0, 1]
    :param xys:
    :param matches_gt:
    """
    img0 = img0.cpu().detach().numpy() if isinstance(img0, torch.Tensor) else img0
    img1 = img1.cpu().detach().numpy() if isinstance(img1, torch.Tensor) else img1
    H, W = img0.shape[0:2]

    w, h = W / 64, H / 64
    pts_x = torch.range(1 / w / 2, 1 - 1 / w / 2, 1 / w)
    pts_y = torch.range(1 / h / 2, 1 - 1 / h / 2, 1 / h)
    pts = torch.stack(torch.meshgrid(pts_y, pts_x), dim=-1).reshape(-1, 2)[..., [1, 0]]

    if img0.dtype is not np.dtype('uint8'):
        img0 = img0 * 255
        img0 = img0.astype(np.uint8)
    if img1.dtype is not np.dtype('uint8'):
        img1 = img1 * 255
        img1 = img1.astype(np.uint8)

    show = np.concatenate([img0, img1], axis=1).copy()  # convert to uint8
    if len(show.shape) == 2 or show.shape[2] == 1:
        show = cv2.cvtColor(show, cv2.COLOR_GRAY2RGB)

    for i in range(len(matches_gt[2])):
        match_id = matches_gt[2][i]
        x0 = int(pts[match_id, 0] * W)
        y0 = int(pts[match_id, 1] * H)
        x1 = int(xys[match_id, 0] * W) + W
        y1 = int(xys[match_id, 1] * H)
        s0 = int(xys[match_id, 2] * 255)
        x2 = int(matches_gt[1][i, 0] * W) + W
        y2 = int(matches_gt[1][i, 1] * H)
        show = cv2.drawMarker(show, (x0, y0), (100, 0, 0), cv2.MARKER_SQUARE, 32)
        show = cv2.line(show, (x0, y0), (x1, y1), (0, s0, 0), 1)
        show = cv2.line(show, (x1, y1), (x2, y2), (0, 0, s0), 1)

    # for i in range(len(matches_gt[3])):
    #     match_id = matches_gt[3][i]
    #     x0 = int(pts[match_id, 0] * W)
    #     y0 = int(pts[match_id, 1] * H)
    #     match_id = matches_gt[3][i]
    #     x1 = int(xys[match_id, 0] * W) + W
    #     y1 = int(xys[match_id, 1] * H)
    #     s0 = int(xys[match_id, 2] * 255)
    #     show = cv2.line(show, (x0, y0), (x1, y1), (s0, 0, 0), 1)

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
    show = show.astype(np.uint8).copy()  # convert to uint8
    if len(show.shape) == 2 or show.shape[2] == 1:
        show = cv2.cvtColor(show, cv2.COLOR_GRAY2RGB)
    h, w, _ = img0.shape
    for i in range(dense_matches.shape[0]):
        m = dense_matches[i]
        cv2.line(show, (int(m[0]) % 8 * 64 + 32, int(m[0]) // 8 * 64 + 32),
                 (int(m[1]) % 8 * 64 + img0.shape[1] + 32, int(m[1]) // 8 * 64 + 32),
                 (0, 255, 0), 1)
    return show


if __name__ == "__main__":
    ''' test plot functions '''
    print("test plot functions")
