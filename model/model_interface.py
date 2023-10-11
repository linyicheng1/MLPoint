import inspect
import torch
import importlib
from torch import Tensor
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from .MLPoint import ML_Point
import utils
from model.losses import match_loss, projection_loss, local_loss


class MInterface(pl.LightningModule):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self.model = ML_Point()
        self.model = self.model.to(self.device)
        if params['pretrained']:
            self.model.load_state_dict(torch.load(params['weight']))

    def forward(self, img: Tensor):
        return self.model(img)

    def loss(self, scores_map_0: Tensor, scores_map_1: Tensor,
             desc_map_00: Tensor, desc_map_01: Tensor,
             desc_map_10: Tensor, desc_map_11: Tensor,
             desc_map_20: Tensor, desc_map_21: Tensor, batch) -> Tensor:

        # 1. gt values
        # warp
        warp01_params = {}
        for k, v in batch['warp01_params'].items():
            warp01_params[k] = v[0]
        warp10_params = {}
        for k, v in batch['warp10_params'].items():
            warp10_params[k] = v[0]
        dense_matches_01 = utils.warp_dense(8, 8, warp01_params)
        dense_matches_10 = utils.warp_dense(8, 8, warp10_params)

        # 2. detection and description
        kps_0 = utils.detection(scores_map_0.detach(), nms_dist=8, threshold=0.1)
        kps_1 = utils.detection(scores_map_1.detach(), nms_dist=8, threshold=0.1)
        # kps
        kps0_valid, kps01_valid, ids01, _ = utils.warp(kps_0, warp01_params)
        kps1_valid, kps10_valid, ids10, _ = utils.warp(kps_1, warp10_params)

        # desc
        sample_pts_0 = kps0_valid * 2. - 1.
        sample_pts_1 = kps1_valid * 2. - 1.
        desc_00 = F.grid_sample(desc_map_00, sample_pts_0.unsqueeze(0).unsqueeze(0),
                                mode='bilinear', align_corners=True, padding_mode='zeros')
        desc_01 = F.grid_sample(desc_map_01, sample_pts_1.unsqueeze(0).unsqueeze(0),
                                mode='bilinear', align_corners=True, padding_mode='zeros')
        desc_10 = F.grid_sample(desc_map_10, sample_pts_0.unsqueeze(0).unsqueeze(0),
                                mode='bilinear', align_corners=True, padding_mode='zeros')
        desc_11 = F.grid_sample(desc_map_11, sample_pts_1.unsqueeze(0).unsqueeze(0),
                                mode='bilinear', align_corners=True, padding_mode='zeros')
        score0 = F.grid_sample(scores_map_0, sample_pts_0.unsqueeze(0).unsqueeze(0),
                                mode='bilinear', align_corners=True, padding_mode='zeros')
        score1 = F.grid_sample(scores_map_1, sample_pts_1.unsqueeze(0).unsqueeze(0),
                                mode='bilinear', align_corners=True, padding_mode='zeros')
        # 3. loss
        # 3.1 projection loss

        loss_kps_0 = projection_loss(kps01_valid.unsqueeze(0), score0.view(1, -1, 1), scores_map_1, 8)
        loss_kps_1 = projection_loss(kps10_valid.unsqueeze(0), score1.view(1, -1, 1), scores_map_0, 8)

        # 3.2 local consistency loss
        loss_desc_00 = local_loss(desc_00.squeeze(2), desc_map_01, kps01_valid.unsqueeze(0), win_size=8)
        loss_desc_01 = local_loss(desc_01.squeeze(2), desc_map_00, kps10_valid.unsqueeze(0), win_size=8)
        loss_desc_10 = local_loss(desc_10.squeeze(2), desc_map_11, kps01_valid.unsqueeze(0), win_size=8)
        loss_desc_11 = local_loss(desc_11.squeeze(2), desc_map_10, kps10_valid.unsqueeze(0), win_size=8)

        # 3.3 dense match loss
        loss_match_01, indices00, indices01 = match_loss(desc_map_20, desc_map_21, dense_matches_01.unsqueeze(0))
        loss_match_10, indices10, indices11 = match_loss(desc_map_21, desc_map_20, dense_matches_10.unsqueeze(0))

        # 3.4 total loss
        proj_weight = self.params['loss']['projection_loss_weight']
        cons_weight = self.params['loss']['local_consistency_loss_weight']
        match_weight = self.params['loss']['dense_matching_loss_weight']
        return proj_weight * (loss_kps_0 + loss_kps_1) + \
               cons_weight * (loss_desc_00 + loss_desc_01 + loss_desc_10 + loss_desc_11) + \
               match_weight * (loss_match_01 + loss_match_10)

    def step(self, batch: Tensor) -> Tensor:
        result0 = self(batch['image0'])
        result1 = self(batch['image1'])
        return self.loss(result0[0], result1[0], result0[1], result1[1],
                         result0[2], result1[2], result0[3], result1[3],
                         batch)

    def training_step(self, batch: Tensor, batch_idx: int) -> STEP_OUTPUT:
        return {"loss": self.step(batch)}

    def configure_optimizers(self):
        if 'weight_decay' in self.params:
            weight_decay = self.params['weight_decay']
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.params['lr'], weight_decay=weight_decay)

        if 'lr_scheduler' not in self.params:
            return optimizer
        else:
            if self.params['lr_scheduler']['type'] == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.params['lr_scheduler']['step_size'],
                                       gamma=self.params['lr_scheduler']['gamma'])
            elif self.params['lr_scheduler']['type'] == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.params['lr_scheduler']['lr_decay_steps'],
                                                  eta_min=self.params['lr_scheduler']['lr_decay_min_lr'])
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]
