import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter


class board(object):
    """
    1. loss curve
        - projection loss
        - local loss 0
        - local loss 1
        - dense matching loss
    2. result image
        - image 0
        - image 1
        - keypoint image 0 [B, 1, H, W]
        - keypoint image 1 [B, 1, H, W]
        - score map [B, 1, H, W]
        - local desc map 0 [B, 3, H, W]
        - dense matching map 0 H/64 x W/64 [B, 3, H, W * 2]
        - dense matching map 1 H/8 x W/8 [B, 3, H, W * 2]
        - sparse matching map [B, 3, H, W * 2]
    3. evaluation metrics
        - Hpatches
    """
    def __init__(self, path):
        self.writer = SummaryWriter(path)

    def close(self):
        self.writer.close()

    ''' 1. loss curve '''
    def add_projection_loss(self, loss_01, loss_10, step):
        self.writer.add_scalar('Loss/projection_01', loss_01, step)
        self.writer.add_scalar('Loss/projection_10', loss_10, step)

    def add_local_loss0(self, loss0, loss1, step):
        self.writer.add_scalar('Loss/local0_0', loss0, step)
        self.writer.add_scalar('Loss/local0_1', loss1, step)

    def add_local_loss1(self, loss0, loss1, step):
        self.writer.add_scalar('Loss/local1_0', loss0, step)
        self.writer.add_scalar('Loss/local1_1', loss1, step)

    def add_dense_matching_loss(self, loss0, loss1, step):
        self.writer.add_scalar('Loss/dense_matching_01', loss0, step)
        self.writer.add_scalar('Loss/dense_matching_10', loss1, step)

    ''' 2. result image '''
    def add_image(self, image0, image1, step):
        self.writer.add_image('Image/0', image0, step, dataformats="NCHW")
        self.writer.add_image('Image/1', image1, step, dataformats="NCHW")

    def add_keypoint_image(self, keypoint0, keypoint1, keypoint01, keypoint10, step):
        if keypoint0 is not None:
            self.writer.add_image('Keypoint/0', keypoint0, step, dataformats="HWC")
        if keypoint1 is not None:
            self.writer.add_image('Keypoint/1', keypoint1, step, dataformats="HWC")
        if keypoint01 is not None:
            self.writer.add_image('Keypoint/01', keypoint01, step, dataformats="HWC")
        if keypoint10 is not None:
            self.writer.add_image('Keypoint/10', keypoint10, step, dataformats="HWC")

    def add_score_map(self, score_map0, score_map1, step):
        if score_map0 is not None:
            self.writer.add_image('Score_map/0', score_map0, step, dataformats="NCHW")
        if score_map1 is not None:
            self.writer.add_image('Score_map/1', score_map1, step, dataformats="NCHW")

    def add_local_desc_map(self, local_desc_map0, local_desc_map1, step):
        if local_desc_map0 is not None:
            self.writer.add_image('Local_desc_map/0', local_desc_map0, step, dataformats="NCHW")
        if local_desc_map1 is not None:
            self.writer.add_image('Local_desc_map/1', local_desc_map1, step, dataformats="NCHW")

    def add_dense_matching_map(self, dense_matching_map0, dense_matching_map1, step):
        self.writer.add_image('Dense_matching_map/0', dense_matching_map0, step)
        self.writer.add_image('Dense_matching_map/1', dense_matching_map1, step)

    def add_sparse_matching_map(self, sparse_matching_map, step):
        self.writer.add_image('Sparse_matching_map', sparse_matching_map, step)

    ''' 3. evaluation metrics '''
    def add_point_metrics(self, num, repeatability, step):
        self.writer.add_scalar('val_kps/num', num, step)
        self.writer.add_scalar('val_kps/repeatability', repeatability, step)

    def add_evaluation_metrics(self, metrics, step):
        self.writer.add_scalar('Evaluation_metrics', metrics, step)


