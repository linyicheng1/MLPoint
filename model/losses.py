import torch
import torch.nn.functional as F


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1).to(scores.device)
    bins1 = alpha.expand(b, 1, n).to(scores.device)
    alpha = alpha.expand(b, 1, 1).to(scores.device)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


def match_loss(desc0, desc1, all_matches, dim=4):
    """
    :param desc0:  (B, D, N)
    :param desc1:  (B, D, N)
    :param all_matches: (B, N, 2)
    :param dim: int
    :return:
    """
    desc0 = desc0.view(1, desc0.shape[1], -1)
    desc1 = desc1.view(1, desc0.shape[1], -1)
    scores = torch.einsum('bdn,bdm->bnm', desc0, desc1)
    # normalize
    dim = desc0.shape[1]
    scores = scores / dim ** 0.5
    # optimal transport
    bin_score = torch.nn.Parameter(torch.tensor(1.))
    scores = log_optimal_transport(scores, bin_score, 100)
    # match loss
    loss = []
    for i in range(len(all_matches[0])):
        x = all_matches[0][i][0]
        y = all_matches[0][i][1]
        loss.append(-torch.log(scores[0][x][y].exp()))
    loss_mean = torch.mean(torch.stack(loss))
    loss_mean = torch.reshape(loss_mean, (1, -1))

    # match result
    # Get the matches with score above "match_threshold".
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    indices0, indices1 = max0.indices, max1.indices
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
    zero = scores.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
    valid0 = mutual0 & (mscores0 > 0.1)
    valid1 = mutual1 & valid0.gather(1, indices1)
    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

    return loss_mean[0], indices0, indices1


def projection_loss(pts01, score, pts1_map, win_size=2):
    """
    :param pts01: (B, N, 3) (x, y, score)
    :param pts1_map: (B, 1, H, W)
    :param score: (B, N, 1)
    :param win_size: int
    :return: loss
    """
    B, N, _ = pts01.shape
    _, _, H, W = pts1_map.shape
    sample_pts = pts01[:, :, 0:2] * 2. - 1.
    score1 = F.grid_sample(pts1_map, sample_pts.unsqueeze(1), mode='bilinear', align_corners=True, padding_mode='zeros')
    loss = (2 - torch.mean(torch.pow(score, 2)) - torch.mean(torch.pow(score1, 2)))
    mask = torch.ones(2*win_size+2, 2*win_size+2).to(pts01.device)
    mask[win_size:win_size+2, win_size:win_size+2] = 0
    for b in range(B):
        for n in range(N):
            # (x, y) in desc1 map
            x, y, s = pts01[b, n, 0], pts01[b, n, 1], score[b, n, 0]
            x0 = int(x * W)
            y0 = int(y * H)
            x1 = x0 + 1
            y1 = y0 + 1
            loss += s * torch.max(pts1_map[b, 0, y0-win_size:y1+win_size+1, x0-win_size:x1+win_size+1] * mask) / N
    return loss


def local_loss(desc0, desc1_map, pts01, win_size=4):
    """
    :param desc0: (B, N, D)
    :param desc1_map:  (B, D, H, W)
    :param pts01: (B, N, 2)
    :param win_size: int
    :return: loss: (B, N)
    """
    B, D, N = desc0.shape
    _, _, H, W = desc1_map.shape
    sample_pts = pts01[:, :, 0:2] * 2. - 1.
    A = (win_size * 2 + 1) ** 2

    desc01 = F.grid_sample(desc1_map, sample_pts.unsqueeze(1), mode='bilinear', align_corners=True, padding_mode='zeros')
    desc01 = desc01.squeeze(2)
    distance = 1 - torch.matmul(desc0[0].T, desc01[0])
    loss = torch.mean(torch.diag(distance))

    # to patch
    patch = torch.nn.Unfold(kernel_size=(win_size * 2 + 1, win_size * 2 + 1), padding=win_size, stride=1)(desc1_map).view(B, D, A, H, W)
    for b in range(B):
        for n in range(N):
            # (x, y) in desc1 map
            x, y = pts01[b, n, 0], pts01[b, n, 1]
            x0 = int(x * W)
            y0 = int(y * H)
            distance = torch.matmul(desc0[b, :, n:n+1].T, patch[0, :, :, y0, x0])
            loss += torch.mean(distance) / N
    return loss



