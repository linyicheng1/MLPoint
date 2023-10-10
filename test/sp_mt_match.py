"""
multi-level matching using superpoint
"""
import cv2
import numpy as np
import torch
from model.SuperPoint import SuperPoint
import utils
from model.losses import match_loss, projection_loss, local_loss


if __name__ == '__main__':
    # 1. define model and load pretrained weights
    model = SuperPoint("../weight/superpoint_v1.pth")

    # 2. load image and preprocess
    img0 = cv2.imread("image3.png", cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread("image4.png", cv2.IMREAD_GRAYSCALE)
    img0 = cv2.resize(img0, (512, 384))
    img1 = cv2.resize(img1, (512, 384))
    img0 = torch.from_numpy(img0).unsqueeze(2).permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0
    img1 = torch.from_numpy(img1).unsqueeze(2).permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0

    # 3. forward
    score_map_0, desc_map_0_0, desc_map_1_0, desc_map_2_0 = model(img0)
    score_map_1, desc_map_0_1, desc_map_1_1, desc_map_2_1 = model(img1)

    # 4. extract keypoints
    kps_0 = utils.detection(score_map_0, nms_dist=8, threshold=0.1)
    kps_1 = utils.detection(score_map_1, nms_dist=8, threshold=0.1)

    # 5. dense matching
    dense_matches = utils.dense_match(desc_map_2_0, desc_map_2_1)

    # 6. prior matching
    # kps_0 = torch.from_numpy(np.array([[0.156, 0.21, 1], [0.2, 0.4, 1], [0.2, 0.6, 1], [0.2, 0.8, 1],
    #                                    [0.4, 0.2, 1], [0.4, 0.4, 1], [0.4, 0.6, 1], [0.4, 0.8, 1],
    #                                    [0.6, 0.2, 1], [0.6, 0.4, 1], [0.6, 0.6, 1], [0.6, 0.8, 1],
    #                                    [0.8, 0.2, 1], [0.8, 0.4, 1], [0.8, 0.6, 1], [0.8, 0.8, 1]])).to(torch.float32)
    # kps_1 = torch.from_numpy(np.array([[0.2, 0.2, 1], [0.2, 0.4, 1], [0.2, 0.6, 1], [0.2, 0.8, 1],
    #                                    [0.4, 0.2, 1], [0.4, 0.4, 1], [0.4, 0.6, 1], [0.4, 0.8, 1],
    #                                    [0.6, 0.2, 1], [0.6, 0.4, 1], [0.6, 0.6, 1], [0.6, 0.8, 1],
    #                                    [0.8, 0.2, 1], [0.8, 0.4, 1], [0.8, 0.6, 1], [0.8, 0.8, 1]])).to(torch.float32)

    prior_matches = utils.prior_match(desc_map_1_0, desc_map_1_1, kps_0, kps_1, dense_matches)

    # 7. optical flow matching
    # flow_kps_0, flow_kps_1 = utils.optical_flow_match(desc_map_0_0, desc_map_0_1, kps_0, kps_1, prior_matches)

    # 8. visualize
    show_kps_0 = utils.plot_keypoints(img0.squeeze(0).permute(1, 2, 0), kps_0)
    show_kps_1 = utils.plot_keypoints(img1.squeeze(0).permute(1, 2, 0), kps_1)
    show_dense_matches = utils.plot_dense_matches(img0.squeeze(0).permute(1, 2, 0),
                                                  img1.squeeze(0).permute(1, 2, 0),
                                                  dense_matches)
    # prior_matches = torch.from_numpy(np.array([[1, 6], [2, 5], [5, 3]])).to(torch.int32)

    show_prior_matches = utils.plot_matches(img0.squeeze(0).permute(1, 2, 0),
                                            img1.squeeze(0).permute(1, 2, 0),
                                            kps_0[:, :-1], kps_1[:, :-1], prior_matches)
    # show_flow_matches = utils.plot_op_matches(img0, img1, flow_kps_0, flow_kps_1)
    cv2.imwrite("show_kps_0.png", show_kps_0)
    cv2.imwrite("show_kps_1.png", show_kps_1)
    cv2.imwrite("show_dense_matches.png", show_dense_matches)
    cv2.imwrite("show_prior_matches.png", show_prior_matches)
    # cv2.imwrite("show_flow_matches.png", show_flow_matches)
    # 9. loss function





