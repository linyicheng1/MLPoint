from model.losses import projection_loss, local_loss, match_loss
import torch

if __name__ == '__main__':
    data = torch.ones((1, 1, 20, 20)) * 0.1
    data[0, 0, 10:12, 10:12] = 0.5
    pts = torch.ones((1, 1, 2)) * 0.5
    score = torch.ones((1, 1, 1)) * 0.5
    loss1 = projection_loss(pts, score, data)
    print("zero loss: ", loss1)

