import torch
from tqdm import tqdm
from data.megadepth import MegaDepthDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model.SuperPoint import SuperPoint
from model.losses import match_loss, projection_loss, local_loss
import utils
import cv2
import torch.nn.functional as F

if __name__ == '__main__':
    torch.manual_seed(3401)
    # read data from MegaDepth dataset
    dataset = MegaDepthDataset(  # root='../data/megadepth',
        root='/home/c211/datasets/megadepth',
        train=True,
        using_cache=True,
        pairs_per_scene=100,
        image_size=512,
        colorjit=True,
        gray=False,
        crop_or_scale='scale',
    )
    # dataset.build_dataset()

    batch_size = 2
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # define model and load pretrained weights
    model = SuperPoint("../weight/superpoint_v1.pth")

    for idx, batch in enumerate(tqdm(loader)):
        # 1. read images
        image0, image1 = batch['image0'], batch['image0']  # [B,3,H,W]

        # 2. model forward
        score_map_0, desc_map_0_0, desc_map_1_0, desc_map_2_0 = model(image0)
        score_map_1, desc_map_0_1, desc_map_1_1, desc_map_2_1 = model(image1)
        kps_0 = utils.detection(score_map_0, nms_dist=8, threshold=0.1)
        kps_1 = utils.detection(score_map_1, nms_dist=8, threshold=0.1)

        # 3. loss function

        # 3.1. gt values
        # warp
        warp01_params = {}
        for k, v in batch['warp01_params'].items():
            warp01_params[k] = v[idx]
        warp10_params = {}
        for k, v in batch['warp10_params'].items():
            warp10_params[k] = v[idx]
        # kps
        kps0_valid, kps01_valid, ids01, _ = utils.warp(kps_0, warp01_params)
        kps1_valid, kps10_valid, ids10, _ = utils.warp(kps_1, warp10_params)

        sample_pts_0 = kps0_valid * 2. - 1.
        sample_pts_1 = kps1_valid * 2. - 1.
        desc_00 = F.grid_sample(desc_map_0_0, sample_pts_0.unsqueeze(0).unsqueeze(0),
                                mode='bilinear', align_corners=True, padding_mode='zeros')
        desc_01 = F.grid_sample(desc_map_0_1, sample_pts_1.unsqueeze(0).unsqueeze(0),
                                mode='bilinear', align_corners=True, padding_mode='zeros')
        desc_10 = F.grid_sample(desc_map_1_0, sample_pts_0.unsqueeze(0).unsqueeze(0),
                                mode='bilinear', align_corners=True, padding_mode='zeros')
        desc_11 = F.grid_sample(desc_map_1_1, sample_pts_1.unsqueeze(0).unsqueeze(0),
                                mode='bilinear', align_corners=True, padding_mode='zeros')
        dense_matches_01 = utils.warp_dense(8, 8, warp01_params)
        dense_matches_10 = utils.warp_dense(8, 8, warp10_params)
        kps01_valid_score = torch.cat([kps01_valid, kps_0[ids01][:, -1:]], dim=1)
        kps10_valid_score = torch.cat([kps10_valid, kps_1[ids10][:, -1:]], dim=1)
        loss_kps_0 = projection_loss(kps01_valid_score.unsqueeze(0), score_map_1, 8)
        loss_kps_1 = projection_loss(kps10_valid_score.unsqueeze(0), score_map_0, 8)
        print(" loss_kps_0 ", loss_kps_0, " loss_kps_1 ", loss_kps_1)
        loss_desc_00 = local_loss(desc_00.squeeze(2), desc_map_0_1, kps01_valid_score.unsqueeze(0), win_size=8)
        loss_desc_01 = local_loss(desc_01.squeeze(2), desc_map_0_0, kps10_valid_score.unsqueeze(0), win_size=8)
        loss_desc_10 = local_loss(desc_10.squeeze(2), desc_map_1_1, kps01_valid_score.unsqueeze(0), win_size=8)
        loss_desc_11 = local_loss(desc_11.squeeze(2), desc_map_1_0, kps10_valid_score.unsqueeze(0), win_size=8)
        print(" loss_desc_00 ", loss_desc_00, " loss_desc_01 ", loss_desc_01,
              " loss_desc_10 ", loss_desc_10, " loss_desc_11 ", loss_desc_11)
        loss_match_01, indices00, indices01 = match_loss(desc_map_2_0, desc_map_2_1, dense_matches_01.unsqueeze(0))
        loss_match_10, indices10, indices11 = match_loss(desc_map_2_1, desc_map_2_0, dense_matches_10.unsqueeze(0))

        # 4. visualization
        # 4.1 plot image
        show_0 = (image0[0].permute(1, 2, 0).numpy() * 255).astype('uint8')
        show_1 = (image1[0].permute(1, 2, 0).numpy() * 255).astype('uint8')
        cv2.cvtColor(show_0, cv2.COLOR_RGB2BGR)
        cv2.cvtColor(show_1, cv2.COLOR_RGB2BGR)
        cv2.imwrite("show_img_0.png", show_0)
        cv2.imwrite("show_img_1.png", show_1)
        # 4.2 plot score map
        show_2 = (score_map_0[0].permute(1, 2, 0).numpy() * 255).astype('uint8')
        show_3 = (score_map_1[0].permute(1, 2, 0).numpy() * 255).astype('uint8')
        show_4 = utils.plot_keypoints(show_3, kps01_valid)
        show_5 = utils.plot_keypoints(show_2, kps10_valid)
        cv2.imwrite("show_score_0.png", show_2)
        cv2.imwrite("show_score_1.png", show_3)
        cv2.imwrite("show_score_01.png", show_4)
        cv2.imwrite("show_score_10.png", show_5)
        # 4.3 plot keypoints
        show_6 = utils.plot_keypoints(show_0, kps0_valid)
        show_7 = utils.plot_keypoints(show_1, kps1_valid)
        show_8 = utils.plot_keypoints(show_1, kps01_valid)
        show_9 = utils.plot_keypoints(show_0, kps10_valid)

        cv2.imwrite("show_kps_0.png", show_6)
        cv2.imwrite("show_kps_1.png", show_7)
        cv2.imwrite("show_kps_01.png", show_8)
        cv2.imwrite("show_kps_10.png", show_9)
        # 4.3 plot dense matches gt
        show_10 = utils.plot_gt_matches(show_0, show_1, dense_matches_01)
        show_11 = utils.plot_gt_matches(show_0, show_1, dense_matches_10)
        cv2.imwrite("show_dense_01.png", show_10)
        cv2.imwrite("show_dense_10.png", show_11)
        # 4.4 plot dense matches pred


        break

