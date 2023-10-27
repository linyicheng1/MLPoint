import cv2
import torch
import torch.nn.functional as F
import numpy as np
from model.losses import log_optimal_transport, arange_like


def compute_dist(desc_0, desc_1, dist_type="dot"):
    """ Compute distance between two descriptors
    :param desc_0: [N, C]
    :param desc_1: [M, C]
    :param dist_type: "dot", "cosine", "l2"
    :return: distance [N, M]
    """
    global distance
    assert dist_type in {"dot", "cosine", "l2"}

    if dist_type == "dot":
        distance = 1 - torch.matmul(desc_0, desc_1.T)
    elif dist_type == "cosine":
        desc_0 = F.normalize(
            desc_0,
            p=2,
            dim=1,
        )
        desc_1 = F.normalize(
            desc_1,
            p=2,
            dim=1,
        )
        distance = 1 - torch.matmul(desc_0, desc_1.T)
    elif dist_type == "l2":
        distance = torch.cdist(desc_0, desc_1, p=2)
    return distance


def match_descriptors(
    distances,
    max_distance=.2,
    cross_check=False,
    max_ratio=0.9,
):
    """ Performs matching of descriptors based on mutual NN distance ratio.
    :param distances: [N, M] distance matrix
    :param max_distance: maximum distance between NN
    :param cross_check: whether to perform cross-check
    :param max_ratio: best/second_best ratio threshold
    :return: matches [K, 2] tensor of matches
    """
    indices1 = torch.arange(distances.shape[0], device=distances.device)
    indices2 = torch.argmin(distances, dim=1)

    if cross_check:
        matches1 = torch.argmin(distances, dim=0)
        mask = indices1 == matches1[indices2]
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if max_distance < torch.inf:
        mask = distances[indices1, indices2] < max_distance
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if max_ratio < 1.0:
        best_distances = distances[indices1, indices2]
        distances[indices1, indices2] = torch.inf
        second_best_indices2 = torch.argmin(distances[indices1], axis=1)
        second_best_distances = distances[indices1, second_best_indices2]
        second_best_distances[second_best_distances == 0] = torch.finfo(
            torch.double
        ).eps
        ratio = best_distances / second_best_distances
        mask = ratio < max_ratio
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    matches = torch.vstack((indices1, indices2))

    return matches.T


def mutual_nearest_neighbor(
    desc_0,
    desc_1,
    distance_fn=compute_dist,
    match_fn=match_descriptors,
    return_distances=False,
):
    """ Performs mutual nearest neighbor matching of descriptors.
    :param desc_0: [N, C] tensor of descriptors
    :param desc_1: [M, C] tensor of descriptors
    :param distance_fn: function that computes distance between descriptors
    :param match_fn: function that performs matching
    :param return_distances: whether to return distances
    :return: matches [K, 2] tensor of matches
    """
    dist = distance_fn(desc_0, desc_1)
    matches = match_fn(dist)
    if return_distances:
        distances = dist[(matches[:, 0], matches[:, 1])]
        return matches, distances
    return matches


def swap_xy(given_ordering, required_ordering, positions):
    assert given_ordering in {"yx", "xy"}
    assert required_ordering in {"yx", "xy"}

    if given_ordering == required_ordering:
        return positions

    return positions[..., [1, 0]]


def ransac(matched_points_0, matched_points_1, ordering="xy"):
    assert len(matched_points_0) == len(matched_points_1)

    if len(matched_points_0) < 4:
        return None

    matched_points_0 = swap_xy(ordering, "xy", matched_points_0)
    matched_points_1 = swap_xy(ordering, "xy", matched_points_1)

    matched_points_0 = matched_points_0.detach().cpu().numpy()
    matched_points_1 = matched_points_1.detach().cpu().numpy()

    estimated_homography, _ = cv2.findHomography(
        matched_points_0,
        matched_points_1,
        cv2.RANSAC,
    )

    if estimated_homography is not None:
        estimated_homography = torch.tensor(
            estimated_homography,
            dtype=torch.float32,
        )

    return estimated_homography


def estimate_homography(
    points_0,
    points_1,
    desc_0,
    desc_1,
    matcher_fn=mutual_nearest_neighbor,
    homography_solver_fn=ransac,
    ordering="xy",
):
    assert ordering in {"xy", "yx"}
    matches = matcher_fn(desc_0, desc_1)
    matched_points_0 = points_0[matches[:, 0]]
    matched_points_1 = points_1[matches[:, 1]]
    estimated_homography = homography_solver_fn(
        matched_points_0[:, :2],
        matched_points_1[:, :2],
        ordering,
    )
    return (
        estimated_homography,
        matched_points_0,
        matched_points_1,
    )


''' Multi-level matching '''


def predict_position(kps, matches, win_size=8):
    position_map_x = ((matches % matches.shape[1]).to(torch.int32)).to(torch.float32)
    position_map_y = ((matches / matches.shape[1]).to(torch.int32)).to(torch.float32)

    grid_x = torch.arange(0, matches.shape[0], 1).to(matches.device)
    grid_y = torch.arange(0, matches.shape[1], 1).to(matches.device)
    grid_y, grid_x = torch.meshgrid(grid_x, grid_y)

    position_map_x[matches < 0] = grid_x[matches < 0].to(torch.float32)
    position_map_y[matches < 0] = grid_y[matches < 0].to(torch.float32)
    position_map_x = position_map_x / matches.shape[1]
    position_map_y = position_map_y / matches.shape[0]
    predict = [[position_map_x[int(kp[1] * matches.shape[0]), int(kp[0] * matches.shape[1])], position_map_y[int(kp[1] * matches.shape[0]), int(kp[0] * matches.shape[1])]] for kp in kps]
    return torch.from_numpy(np.array(predict))


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


def dense_match(desc_map_0: torch.tensor, desc_map_1: torch.tensor) -> torch.tensor:
    """
    :param desc_map_0: [1, C, H/64, W/64] dense descriptor map of image 0
    :param desc_map_1: [1, C, H/64, W/64] dense descriptor map of image 1
    :return: [2, H/64, W/64] dense matches map [0->1, 1->0]
    """

    B, D, H, W = desc_map_0.shape
    desc0 = desc_map_0.view(1, D, -1)
    desc1 = desc_map_1.view(1, D, -1)
    scores = torch.einsum('bdn,bdm->bnm', desc0, desc1)

    # match result
    # Get the matches with score above "match_threshold".
    max0, max1 = scores.max(2), scores.max(1)
    indices0, indices1 = max0.indices, max1.indices
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
    zero = scores.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
    valid0 = mutual0 & (mscores0 > 0.0)
    valid1 = mutual1 & valid0.gather(1, indices1)
    # indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    # indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
    return torch.stack([indices0.view(H, W), indices1.view(H, W)], dim=0)


def prior_match(
        desc_map_0: torch.tensor, desc_map_1: torch.tensor,
        kps_0: torch.tensor, kps_1: torch.tensor,
        matches: torch.tensor, max_distance=torch.inf, max_ratio=1.0):
    sample_pts_0 = kps_0[:, :-1].unsqueeze(0).unsqueeze(0) * 2 - 1
    sample_pts_1 = kps_1[:, :-1].unsqueeze(0).unsqueeze(0) * 2 - 1
    desc_0 = torch.nn.functional.grid_sample(desc_map_0, sample_pts_0, mode='bilinear', padding_mode='border')
    desc_1 = torch.nn.functional.grid_sample(desc_map_1, sample_pts_1, mode='bilinear', padding_mode='border')
    desc_0 = desc_0.squeeze(0).permute(1, 2, 0).squeeze(0)
    desc_1 = desc_1.squeeze(0).permute(1, 2, 0).squeeze(0)

    # return mutual_nearest_neighbor(desc_0, desc_1)
    pos_01 = predict_position(kps_0.cpu(), matches[0].cpu()).to(kps_0.device)
    pos_10 = predict_position(kps_1.cpu(), matches[1].cpu()).to(kps_0.device)
    desc_0 = torch.cat([desc_0, pos_01 * 0.5], dim=1).to(torch.float32)
    desc_1 = torch.cat([desc_1, kps_1[:, :-1] * 0.5], dim=1).to(torch.float32)
    dist = compute_dist(desc_0, desc_1)
    matches01 = match_descriptors(dist, max_distance, True, max_ratio)
    desc_0 = torch.cat([desc_0, kps_0[:, :-1] * 0.5], dim=1).to(torch.float32)
    desc_1 = torch.cat([desc_1, pos_10 * 0.5], dim=1).to(torch.float32)
    dist = compute_dist(desc_0, desc_1)
    matches10 = match_descriptors(dist, max_distance, True, max_ratio)
    result = torch.cat([matches01, matches10], dim=0)
    return torch.unique(result, dim=0)


def optical_flow_match(
        desc_map_0: torch.tensor, desc_map_1: torch.tensor,
        kps_0: torch.tensor, kps_1: torch.tensor,
        matches: torch.tensor, win_size: int = 8):
    return torch.zeros(0)
    pass


def ml_match(
    desc_map_00: torch.tensor, desc_map_01: torch.tensor,
    desc_map_10: torch.tensor, desc_map_11: torch.tensor,
    desc_map_20: torch.tensor, desc_map_21: torch.tensor,
    kps_0: torch.tensor, kps_1: torch.tensor
):
    """
    :param desc_map_00: [C1, H/64, W/64]
    :param desc_map_01: [C1, H/64, W/64]
    :param desc_map_10: [C2, H/8, W/8]
    :param desc_map_11: [C2, H/8, W/8]
    :param desc_map_20: [C3, H, W]
    :param desc_map_21: [C3, H, W]
    :param kps_0: [M, 2]
    :param kps_1: [N, 2]
    :return: [K, 2]
    """
    matches0 = dense_match(desc_map_20, desc_map_21)
    matches1 = prior_match(desc_map_10, desc_map_11, kps_0, kps_1, matches0)
    # matches2 = optical_flow_match(desc_map_00, desc_map_01, kps_0, kps_1, matches1)
    return matches0, matches1, None



