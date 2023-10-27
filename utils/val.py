import cv2
import torch
import utils
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


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


def val_matches(
        show0, show1,
        desc_map_00: torch.tensor, desc_map_01: torch.tensor,
        desc_map_10: torch.tensor, desc_map_11: torch.tensor,
        desc_map_20: torch.tensor, desc_map_21: torch.tensor,
        kps_0: torch.tensor, kps_1: torch.tensor,
        warp01, warp10, th: int = 3):
    w = show0.shape[1]
    matches0, matches1, matches2 = utils.ml_match(desc_map_00, desc_map_01,
                                                  desc_map_10, desc_map_11,
                                                  desc_map_20, desc_map_21,
                                                  kps_0, kps_1)
    img0 = utils.plot_nn_matches(show0, show1, matches0)

    m_pts_0 = kps_0[matches1[:, 0], :-1]
    m_pts_1 = kps_1[matches1[:, 1], :-1]
    num_putative = len(m_pts_0)

    mkpts0, mkpts01, ids0, _ = utils.warp(m_pts_0, warp01)
    mkpts1 = m_pts_1[ids0]

    dist = torch.sqrt(((mkpts01 - mkpts1) ** 2).sum(axis=1)).cpu()
    if dist.shape[0] == 0:
        dist = dist.new_tensor([float('inf')])

    num_inlier = sum(dist*512 <= th)

    img1 = utils.plot_op_matches(show0, show1, mkpts0, mkpts1, dist*512 <= th)

    cv2.imwrite("img0.png", img0)
    cv2.imwrite("img1.png", img1)
    return {
        "num_cov_feat": num_putative,
        "num_inlier": num_inlier,
    }


def load_precompute_errors(file):
    errors = torch.load(file)
    return errors


def draw_MMA(errors):
    plt.switch_backend('agg')
    methods = ['hesaff', 'hesaffnet', 'delf', 'superpoint', 'lf-net', 'd2-net-trained', 'd2-net-trained-ms']
    names = ['Hes. Aff. + Root-SIFT', 'HAN + HN++', 'DELF', 'SuperPoint', 'LF-Net', 'D2-Net Trained',
             'D2-Net Trained MS']
    colors = ['red', 'orange', 'olive', 'green', 'blue', 'brown', 'purple']
    linestyles = ['--', '--', '--', '--', '--', '--', '--']

    methods += ['r2d2_WASF_N16.scale-0.3-1', 'disk', ]
    names += ['r2d2', 'disk', ]
    colors += ['silver', 'sandybrown', ]
    linestyles += ['--', '--', ]

    # methods += ['ours']
    # names += ['ours']
    # colors += ['cyan']
    # linestyles += ['-']

    n_i = 52
    n_v = 56

    plt_lim = [1, 10]
    plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)

    plt.rc('axes', titlesize=25)
    plt.rc('axes', labelsize=25)

    fig = plt.figure(figsize=(20, 5))
    canvas = FigureCanvas(fig)

    plt.subplot(1, 4, 1)
    for method, name, color, ls in zip(methods, names, colors, linestyles):
        i_err, v_err, _ = errors[method]
        plt.plot(plt_rng, [(i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5) for thr in plt_rng], color=color, ls=ls,
                 linewidth=3, label=name)
    plt.title('Overall')
    plt.xlim(plt_lim)
    plt.xticks(plt_rng)
    plt.ylabel('MMA')
    plt.ylim([0, 1])
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend()

    plt.subplot(1, 4, 2)
    for method, name, color, ls in zip(methods, names, colors, linestyles):
        i_err, v_err, _ = errors[method]
        plt.plot(plt_rng, [i_err[thr] / (n_i * 5) for thr in plt_rng], color=color, ls=ls, linewidth=3, label=name)
    plt.title('Illumination')
    # plt.xlabel('threshold [px]')
    plt.xlim(plt_lim)
    plt.xticks(plt_rng)
    plt.ylim([0, 1])
    plt.gca().axes.set_yticklabels([])
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.subplot(1, 4, 3)
    for method, name, color, ls in zip(methods, names, colors, linestyles):
        i_err, v_err, _ = errors[method]
        plt.plot(plt_rng, [v_err[thr] / (n_v * 5) for thr in plt_rng], color=color, ls=ls, linewidth=3, label=name)
    plt.title('Viewpoint')
    plt.xlim(plt_lim)
    plt.xticks(plt_rng)
    plt.ylim([0, 1])
    plt.gca().axes.set_yticklabels([])
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.subplot(1, 4, 4)

    canvas.draw()  # draw the canvas, cache the renderer
    width, height = canvas.get_width_height()

    # Option 2a: Convert to a NumPy array.
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    plt.close()

    return image[:, 150:1420, :]


if __name__ == '__main__':
    errors = load_precompute_errors('errors.pkl')
    image = draw_MMA(errors)
    plt.imshow(image), plt.savefig("mma.png")
