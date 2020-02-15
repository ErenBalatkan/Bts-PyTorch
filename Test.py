
import KittiDataLoader
import BTS
import torch
import cv2
import numpy as np
from DepthVisualizer import DepthRenderer
import argparse
import sys
import os
import configs

model_path = configs.MODEL_PATH
dataset_path = configs.DATASET_PATH

MAKE_VIDEO = configs.MAKE_VIDEO
video_save_path = configs.VIDEO_SAVE_PATH

DISPLAY_VIDEO = configs.DISPLAY_VIDEO

dataloader = KittiDataLoader.KittiDataset(dataset_path, is_test=True)
model = BTS.BtsController()
model.load_model(model_path)
model.eval()


def Unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


loss_names = ["Silog", "rmse", "rmse_log", "abs_rel", "sq_rel"]  # This is used for printing
def CalculateLosses(pred, gt):
    Silog = BTS.SilogLoss()
    silog_loss = Silog(pred, gt, 100, 1)

    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    # Filtering, similar to the one used in official implementation
    mask = torch.tensor((gt > 1e-3) & (gt < 80), dtype=torch.bool)
    masked_pred = torch.masked_select(pred, mask)
    masked_gt = torch.masked_select(gt, mask)

    masked_pred[masked_pred < 1e-3] = 1e-3
    masked_pred[masked_pred > 80] = 80

    rmse = torch.sqrt(torch.mean((masked_gt-masked_pred)**2)).item()
    rmse_log = torch.sqrt(((torch.log(masked_gt) - torch.log(masked_pred))**2).mean()).item()
    abs_rel = torch.mean(torch.abs(masked_gt-masked_pred) / masked_gt).item()
    sq_rel = torch.mean((masked_gt-masked_pred)**2 / masked_gt).item()

    return [silog_loss, rmse, rmse_log, abs_rel, sq_rel]


VIDEO_RES = (1600, 1200)  #(1216, 352 * 2)
if MAKE_VIDEO:
    video = cv2.VideoWriter('video2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, VIDEO_RES)


def predict(model, input, is_channels_first=True, focal=715.0873):
    if is_channels_first:
        tensor_input = torch.tensor(input).unsqueeze(-1).to("cuda").float().transpose(0, 3).transpose(2, 3).transpose(1, 2)
    else:
        tensor_input = torch.tensor(input).unsqueeze(-1).to("cuda").float().transpose(0, 3).transpose(1, 2).transpose(2, 3)

    model_output = model(tensor_input, torch.tensor(focal).unsqueeze(0))[-1][0].detach().cpu().transpose(0, 1).transpose(1, 2).squeeze(-1)
    return model_output

losses = []

last_idx = 0
for idx, item in enumerate(dataloader):
    if (item is None):
        break

    result_raw = model.predict(item["image"], item["focal_length"])
    last_idx = idx
    label = item["label"][0]
    losses += [CalculateLosses(result_raw, label)]

    im_vis = np.asarray((Unnormalize(item["image"])*255).transpose(0, 1).transpose(1, 2), np.uint8)
    # point_cloud = renderer.convert_depthmap_to_points(label, 1 * item["focal_length"].cpu().numpy(),
    #                                                   im_vis)
    # renderer.set_points(point_cloud)
    # renderer.render(True)

    result_vis = model.depth_map_to_rgbimg(result_raw)
    result = np.append(result_vis, im_vis, axis=0)
    # result = im_vis

    # result = renderer.get_rendered_frame()
    if MAKE_VIDEO:
        video.write(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    if DISPLAY_VIDEO:
        cv2.imshow("a", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

if MAKE_VIDEO:
    video.release()

cv2.destroyAllWindows()

losses = np.array(losses)
for i in range(len(loss_names)):
    print(loss_names[i], ":", np.mean(losses[:, i]))
