import inspect
import logging

import cv2

from utils.logger import board
import torch
import importlib
import numpy as np
from torch import Tensor
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from .MLPoint import ML_Point
import utils
from model.losses import match_loss, projection_loss, local_loss
from model.SuperPoint import SuperPoint


class MInterface(pl.LightningModule):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self.model = ML_Point(params['model_size'])
        # self.sp = SuperPoint(params['weight'])
        self.model = self.model.to(self.device)
        if params['pretrained']:
            self.model.load_state_dict(torch.load(params['weight']))
        self.board = board("lightning_logs/logs")
        self.losses = {}
        # vals
        self.num_feat = None
        self.repeatability = None
        self.accuracy = None
        self.matching_score = None
        self.max_repeatability = 0

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
        # val_pts = utils.val_key_points(kps_0, kps_1, warp01_params, warp10_params)
        # self.num_feat = val_pts['num_feat']
        # self.repeatability = val_pts['repeatability']
        # kps
        kps0_valid, kps01_valid, ids01, ids01_out = utils.warp(kps_0, warp01_params)
        kps1_valid, kps10_valid, ids10, ids10_out = utils.warp(kps_1, warp10_params)

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
        sample_pts_0_out = kps_0[ids01_out][:, :-1] * 2. - 1.
        sample_pts_1_out = kps_1[ids10_out][:, :-1] * 2. - 1.
        desc_00_out = F.grid_sample(desc_map_00, sample_pts_0_out.unsqueeze(0).unsqueeze(0),
                                    mode='bilinear', align_corners=True, padding_mode='zeros')
        desc_01_out = F.grid_sample(desc_map_01, sample_pts_1_out.unsqueeze(0).unsqueeze(0),
                                    mode='bilinear', align_corners=True, padding_mode='zeros')
        desc_10_out = F.grid_sample(desc_map_10, sample_pts_0_out.unsqueeze(0).unsqueeze(0),
                                    mode='bilinear', align_corners=True, padding_mode='zeros')
        desc_11_out = F.grid_sample(desc_map_11, sample_pts_1_out.unsqueeze(0).unsqueeze(0),
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
        loss_desc_00 = local_loss(desc_00.squeeze(2).transpose(1, 2),
                                  desc_00_out.squeeze(2).transpose(1, 2),
                                  desc_map_01, kps01_valid.unsqueeze(0), win_size=8)
        loss_desc_01 = local_loss(desc_01.squeeze(2).transpose(1, 2),
                                  desc_01_out.squeeze(2).transpose(1, 2),
                                  desc_map_00, kps10_valid.unsqueeze(0), win_size=8)

        # loss_desc_10 = local_loss(desc_10.squeeze(2), desc_map_11, kps01_valid.unsqueeze(0), win_size=8)
        # loss_desc_11 = local_loss(desc_11.squeeze(2), desc_map_10, kps10_valid.unsqueeze(0), win_size=8)

        # 3.3 dense match loss
        # loss_match_01, indices00, indices01 = match_loss(desc_map_20, desc_map_21, dense_matches_01.unsqueeze(0))
        # loss_match_10, indices10, indices11 = match_loss(desc_map_21, desc_map_20, dense_matches_10.unsqueeze(0))

        # 3.4 total loss
        proj_weight = 1 # self.params['loss']['projection_loss_weight']
        cons_weight = self.params['loss']['local_consistency_loss_weight']
        match_weight = self.params['loss']['dense_matching_loss_weight']

        self.losses = {
            'image0': batch['image0'],
            'image1': batch['image1'],
            'scores_map_0': scores_map_0,
            'scores_map_1': scores_map_1,
            'loss_kps_0': loss_kps_0,
            'loss_kps_1': loss_kps_1,
            'kps_0': kps_0[:, :2],
            'kps_1': kps_1[:, :2],
            'kps_01': kps01_valid,
            'kps_10': kps10_valid,
            # 'loss_desc_00': loss_desc_00,
            # 'loss_desc_01': loss_desc_01,
            'desc_map_00': desc_map_00,
            'desc_map_01': desc_map_01,
            # 'loss_desc_10': loss_desc_10,
            # 'loss_desc_11': loss_desc_11,
            # 'loss_match_01': loss_match_01,
            # 'loss_match_10': loss_match_10
        }

        return proj_weight * (loss_kps_0 + loss_kps_1) + cons_weight * (loss_desc_00 + loss_desc_01)
               # match_weight * (loss_match_01 + loss_match_10)  # + \


    def step(self, batch: Tensor) -> Tensor:
        result0 = self(batch['image0'])
        result1 = self(batch['image1'])
        loss = self.loss(result0[0], result1[0], result0[1], result1[1],
                         result0[2], result1[2], result0[3], result1[3],
                         batch)
        return loss

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

    def backward(self, loss: Tensor, *args, **kwargs) -> None:
        if loss.requires_grad:
            loss.backward()
        else:
            logging.debug('loss does not require grad, skipping backward')

    def on_after_backward(self) -> None:
        if  self.global_step % 10 == 0:
            # log
            # self.board.add_local_loss0(self.losses['loss_desc_00'], self.losses['loss_desc_01'], self.global_step)
            self.board.add_projection_loss(self.losses['loss_kps_0'], self.losses['loss_kps_1'], self.global_step)
            # self.board.add_dense_matching_loss(self.losses['loss_match_01'], self.losses['loss_match_10'], self.global_step)
            # self.board.add_image(self.losses['image0'], self.losses['image1'], self.global_step)
            self.board.add_score_map(self.losses['scores_map_0'], self.losses['scores_map_1'], self.global_step)
            show_0 = (self.losses['image0'][0].cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
            show_1 = (self.losses['image1'][0].cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
            show_6 = utils.plot_keypoints(show_0, self.losses['kps_0'].cpu())
            show_7 = utils.plot_keypoints(show_1, self.losses['kps_1'].cpu())
            # show_8 = utils.plot_keypoints(show_1, self.losses['kps_01'].cpu())
            # show_9 = utils.plot_keypoints(show_0, self.losses['kps_10'].cpu())
            # self.board.add_local_desc_map(self.losses['desc_map_00'], self.losses['desc_map_01'], self.global_step)
            self.board.add_keypoint_image(show_6, show_7, None, None, self.global_step)
            # self.board.add_point_metrics(self.num_feat, self.repeatability)
        # 100 times save model
        if self.global_step % 100 == 0:
            torch.save(self.model.state_dict(), "/home/server/linyicheng/py_proj/MLPoint/weight/last_{}.pth".format(self.global_step))

    def validation_step(self, batch: Tensor, batch_idx: int) -> STEP_OUTPUT:
        warp01_params = {}
        for k, v in batch['warp01_params'].items():
            warp01_params[k] = v[0]
        warp10_params = {}
        for k, v in batch['warp10_params'].items():
            warp10_params[k] = v[0]
        self.model.eval()
        result0 = self(batch['image0'])
        result1 = self(batch['image1'])
        nms_dist = self.params['nms_dist']
        threshold = self.params['threshold']
        top_k = self.params['top_k']
        min_score = self.params['min_score']
        # sp
        # result0 = self.sp(batch['image0'].cpu())
        # result1 = self.sp(batch['image1'].cpu())
        # harris
        # img0 = (torch.sum(batch['image0'][0].cpu().permute(1, 2, 0), dim=2) * 255).numpy().astype('uint8')
        # img1 = (torch.sum(batch['image1'][0].cpu().permute(1, 2, 0), dim=2) * 255).numpy().astype('uint8')
        #
        # harris_map_0 = torch.from_numpy(cv2.cornerHarris(img0, 2, 3, 0.04)).unsqueeze(0).unsqueeze(0)
        # harris_map_1 = torch.from_numpy(cv2.cornerHarris(img1, 2, 3, 0.04)).unsqueeze(0).unsqueeze(0)
        # kps_0 = utils.detection(harris_map_0.to(batch['image0'].device), nms_dist=4, threshold=0.0, max_pts=1000)
        # kps_1 = utils.detection(harris_map_1.to(batch['image0'].device), nms_dist=4, threshold=0.0, max_pts=1000)

        kps_0 = utils.detection(result0[0].detach().to(batch['image0'].device), nms_dist=nms_dist, threshold=min_score,
                                max_pts=top_k)
        kps_1 = utils.detection(result1[0].detach().to(batch['image0'].device), nms_dist=nms_dist, threshold=min_score,
                                max_pts=top_k)

        val_pts = utils.val_key_points(kps_0, kps_1, warp01_params, warp10_params, th=threshold)
        if val_pts['num_feat'] > 0:
            self.num_feat.append(val_pts['num_feat'])
            self.repeatability.append(val_pts['repeatability'])
        # if val_pts['repeatability'] < 0.3:
        #     show0 = (batch['image0'][0].cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
        #     show1 = (batch['image1'][0].cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
        #     kps0_cov, kps01_cov, _, _ = utils.warp(kps_0, warp01_params)
        #     kps1_cov, kps10_cov, _, _ = utils.warp(kps_1, warp10_params)
        #     show6 = utils.plot_keypoints(show0, kps_0.cpu())
        #     show7 = utils.plot_keypoints(show1, kps_1.cpu())
        #     show8 = utils.plot_keypoints(show1, kps01_cov.cpu())
        #     show9 = utils.plot_keypoints(show0, kps10_cov.cpu())
        # cv2.imwrite('lightning_logs/{}_0.jpg'.format(batch_idx), show6)
        # cv2.imwrite('lightning_logs/{}_1.jpg'.format(batch_idx), show7)
        # cv2.imwrite('lightning_logs/{}_01.jpg'.format(batch_idx), show8)
        # cv2.imwrite('lightning_logs/{}_10.jpg'.format(batch_idx), show9)
        # print("hello")

        # print('step: ', batch_idx, 'num_feat: ', val_pts['num_feat'], 'repeatability: ', val_pts['repeatability'])

        return {"num_feat_mean": 0,
                "repeatability_mean": 0,
                }

    def on_validation_start(self):
        self.num_feat = []
        self.repeatability = []
        self.accuracy = []
        self.matching_score = []

    def on_validation_end(self) -> None:
        num_feat_mean = np.mean(np.array(self.num_feat))
        repeatability_mean = np.mean(np.array(self.repeatability))
        if self.max_repeatability < repeatability_mean:
            self.max_repeatability = repeatability_mean
            torch.save(self.model.state_dict(),
                       "/home/server/linyicheng/py_proj/MLPoint/weight/best_{}.pth".format(repeatability_mean))

        print('num_feat_mean: ', num_feat_mean)
        print('repeatability_mean: ', repeatability_mean)
        self.board.add_point_metrics(num_feat_mean, repeatability_mean, self.global_step)

    def on_test_start(self) -> None:
        self.num_feat = []
        self.repeatability = []
        self.accuracy = []
        self.matching_score = []

    def on_test_end(self) -> None:
        num_feat_mean = np.mean(np.array(self.num_feat))
        repeatability_mean = np.mean(np.array(self.repeatability))
        print('num_feat_mean: ', num_feat_mean)
        print('repeatability_mean: ', repeatability_mean)
        self.board.add_point_metrics(num_feat_mean, repeatability_mean, self.global_step)

    def test_step(self, batch: Tensor, batch_idx: int) -> STEP_OUTPUT:
        warp01_params = {}
        for k, v in batch['warp01_params'].items():
            warp01_params[k] = v[0]
        warp10_params = {}
        for k, v in batch['warp10_params'].items():
            warp10_params[k] = v[0]
        self.model.eval()
        result0 = self(batch['image0'])
        result1 = self(batch['image1'])
        nms_dist = self.params['nms_dist']
        threshold = self.params['threshold']
        top_k = self.params['top_k']
        min_score = self.params['min_score']
        # sp
        # result0 = self.sp(batch['image0'].cpu())
        # result1 = self.sp(batch['image1'].cpu())
        # harris
        # img0 = (torch.sum(batch['image0'][0].cpu().permute(1, 2, 0), dim=2) * 255).numpy().astype('uint8')
        # img1 = (torch.sum(batch['image1'][0].cpu().permute(1, 2, 0), dim=2) * 255).numpy().astype('uint8')
        #
        # harris_map_0 = torch.from_numpy(cv2.cornerHarris(img0, 2, 3, 0.04)).unsqueeze(0).unsqueeze(0)
        # harris_map_1 = torch.from_numpy(cv2.cornerHarris(img1, 2, 3, 0.04)).unsqueeze(0).unsqueeze(0)
        # kps_0 = utils.detection(harris_map_0.to(batch['image0'].device), nms_dist=4, threshold=0.0, max_pts=1000)
        # kps_1 = utils.detection(harris_map_1.to(batch['image0'].device), nms_dist=4, threshold=0.0, max_pts=1000)

        kps_0 = utils.detection(result0[0].detach().to(batch['image0'].device), nms_dist=nms_dist, threshold=min_score, max_pts=top_k)
        kps_1 = utils.detection(result1[0].detach().to(batch['image0'].device), nms_dist=nms_dist, threshold=min_score, max_pts=top_k)

        val_pts = utils.val_key_points(kps_0, kps_1, warp01_params, warp10_params, th=threshold)
        if val_pts['num_feat'] > 0:
            self.num_feat.append(val_pts['num_feat'])
            self.repeatability.append(val_pts['repeatability'])
        # if val_pts['repeatability'] < 0.3:
        #     show0 = (batch['image0'][0].cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
        #     show1 = (batch['image1'][0].cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
        #     kps0_cov, kps01_cov, _, _ = utils.warp(kps_0, warp01_params)
        #     kps1_cov, kps10_cov, _, _ = utils.warp(kps_1, warp10_params)
        #     show6 = utils.plot_keypoints(show0, kps_0.cpu())
        #     show7 = utils.plot_keypoints(show1, kps_1.cpu())
        #     show8 = utils.plot_keypoints(show1, kps01_cov.cpu())
        #     show9 = utils.plot_keypoints(show0, kps10_cov.cpu())
            # cv2.imwrite('lightning_logs/{}_0.jpg'.format(batch_idx), show6)
            # cv2.imwrite('lightning_logs/{}_1.jpg'.format(batch_idx), show7)
            # cv2.imwrite('lightning_logs/{}_01.jpg'.format(batch_idx), show8)
            # cv2.imwrite('lightning_logs/{}_10.jpg'.format(batch_idx), show9)
            # print("hello")

        print('step: ', batch_idx, 'num_feat: ', val_pts['num_feat'], 'repeatability: ', val_pts['repeatability'])

        return {"num_feat_mean": 0,
                "repeatability_mean": 0,
                }

