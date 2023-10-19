import torch
import utils


def mutual_argmax(value, mask=None, as_tuple=True):
    """
    Args:
        value: MxN
        mask:  MxN

    Returns:

    """
    value = value - value.min()  # convert to non-negative tensor
    if mask is not None:
        value = value * mask

    max0 = value.max(dim=1, keepdim=True)  # the col index the max value in each row
    max1 = value.max(dim=0, keepdim=True)

    valid_max0 = value == max0[0]
    valid_max1 = value == max1[0]

    mutual = valid_max0 * valid_max1
    if mask is not None:
        mutual = mutual * mask

    return mutual.nonzero(as_tuple=as_tuple)


def mutual_argmin(value, mask=None):
    return mutual_argmax(-value, mask)


def compute_keypoints_distance(kpts0, kpts1, p=2):
    """
    Args:
        kpts0: torch.tensor [M,2]
        kpts1: torch.tensor [N,2]
        p: (int, float, inf, -inf, 'fro', 'nuc', optional): the order of norm

    Returns:
        dist, torch.tensor [N,M]
    """
    dist = kpts0[:, None, :] - kpts1[None, :, :]  # [M,N,2]
    dist = torch.norm(dist, p=p, dim=2)  # [M,N]
    return dist


def val_key_points(kps0, kps1, warp01, warp10, th: int = 3):
    num_feat = min(kps0.shape[0], kps1.shape[0])

    # ==================================== covisible keypoints
    kps0_cov, kps01_cov, _, _ = utils.warp(kps0, warp01)
    kps1_cov, kps10_cov, _, _ = utils.warp(kps1, warp10)
    num_cov_feat = (len(kps0_cov) + len(kps1_cov)) / 2  # number of covisible keypoints
    if kps0_cov.shape[0] == 0 or kps1_cov.shape[0] == 0:
        return {
            'num_feat': 0,
            'repeatability': 0,
        }
    # ==================================== get gt matching keypoints
    dist01 = compute_keypoints_distance(kps0_cov, kps10_cov)
    dist10 = compute_keypoints_distance(kps1_cov, kps01_cov)
    dist_mutual = (dist01 + dist10.t()) / 2.
    imutual = torch.arange(min(dist_mutual.shape), device=dist_mutual.device)
    dist_mutual[imutual, imutual] = 99999  # mask out diagonal
    mutual_min_indices = mutual_argmin(dist_mutual)
    dist = dist_mutual[mutual_min_indices]
    if 'resize' in warp01:
        dist = dist * warp01['resize']
    else:
        dist = dist * warp01['width']
    gt_num = (dist <= th).sum().cpu()  # number of gt matching keypoints

    return {
        'num_feat': num_feat,
        'repeatability': gt_num / num_feat,
    }

