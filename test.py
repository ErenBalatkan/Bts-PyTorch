
import KittiDataLoader
import BTS
import os
import torch
import cv2
from matplotlib import pyplot as plt
import numpy as np

model_path = "models/Balatkan/bts_model_epoch_49"
dataset_path = "e://Code/Tez/bts_eren/kitti"

MAKE_VIDEO = True
video_save_path = "video.avi"

DISPLAY_VIDEO = True


dataloader = KittiDataLoader.KittiDataset(dataset_path, is_test=True)
model = BTS.BtsController()
model.load_model(model_path)




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
    mask = (gt > 1e-3) & (gt < 80)
    masked_pred = torch.masked_select(pred, mask)
    masked_gt = torch.masked_select(gt, mask)

    masked_pred[masked_pred < 1e-3] = 1e-3
    masked_pred[masked_pred > 80] = 80

    rmse = torch.sqrt(torch.mean((masked_gt-masked_pred)**2)).item()
    rmse_log = torch.sqrt(((torch.log(masked_gt) - torch.log(masked_pred))**2).mean()).item()
    abs_rel = torch.mean(torch.abs(masked_gt-masked_pred) / masked_gt).item()
    sq_rel = torch.mean((masked_gt-masked_pred)**2 / masked_gt).item()

    return [silog_loss, rmse, rmse_log, abs_rel, sq_rel]


if MAKE_VIDEO:
    video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (1216, 352 * 2))

losses = []

last_idx = 0
for idx, item in enumerate(dataloader):
    if (item is None):
        break
    result_raw = model.predict(item["image"])

    last_idx = idx
    label = item["label"][0]
    losses += [CalculateLosses(result_raw, label)]

    result_vis = model.depth_map_to_rgbimg(result_raw)
    im_vis = np.asarray((Unnormalize(item["image"])*255).transpose(0, 1).transpose(1, 2), np.uint8)
    result = np.append(result_vis, im_vis, axis=0)
    if MAKE_VIDEO:
        video.write(result)
    if DISPLAY_VIDEO:
        cv2.imshow("a", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

if MAKE_VIDEO:
    video.release()

cv2.destroyAllWindows()

losses = np.array(losses)
for i in range(len(loss_names)):
    print(loss_names[i], ":", np.mean(losses[:, i]))
