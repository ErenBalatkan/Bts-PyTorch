import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torch import optim
import os
import math
import cv2
import albumentations as A

from torch.utils.tensorboard import SummaryWriter

activation_fn = nn.ELU()

MAX_DEPTH = 81
DEPTH_OFFSET = 0.1 # This is used for ensuring depth prediction gets into positive range

USE_APEX = False  # Enable if you have GPU with Tensor Cores, otherwise doesnt really bring any benefits.
APEX_OPT_LEVEL = "O2"

BATCH_NORM_MOMENTUM = 0.005
ENABLE_BIAS = True

device = torch.device("cpu")
if torch.cuda.is_available() :
    device = torch.device("cuda")

if USE_APEX:
    import apex


class UpscaleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpscaleLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=ENABLE_BIAS)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_MOMENTUM)

    def forward(self, input):
        input = nn.functional.interpolate(input, scale_factor=2, mode="nearest")
        input = activation_fn(self.conv(input))
        input = self.bn(input)
        return input


class UpscaleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpscaleBlock, self).__init__()
        self.uplayer = UpscaleLayer(in_channels, out_channels)
        self.conv = nn.Conv2d(out_channels+skip_channels, out_channels, 3, padding=1, bias=ENABLE_BIAS)
        self.bn2 = nn.BatchNorm2d(out_channels, BATCH_NORM_MOMENTUM)

    def forward(self, input_j):
        input, skip = input_j
        input = self.uplayer(input)
        cat = torch.cat((input, skip), 1)
        input = activation_fn(self.conv(cat))
        input = self.bn2(input)
        return input, cat


class UpscaleNetwork(nn.Module):
    def __init__(self, filters=[512, 256]):
        super(UpscaleNetwork, self,).__init__()
        self.upscale_block1 = UpscaleBlock(2208, 384, filters[0])  # H16
        self.upscale_block2 = UpscaleBlock(filters[0], 192, filters[1])  # H8

    def forward(self, raw_input):
        input, h2, h4, h8, h16 = raw_input
        input, _ = self.upscale_block1((input, h16))
        input, cat = self.upscale_block2((input, h8))
        return input, cat


class AtrousBlock(nn.Module):
    def __init__(self, input_filters, filters, dilation, apply_initial_bn=True):
        super(AtrousBlock, self).__init__()

        self.initial_bn = nn.BatchNorm2d(input_filters, BATCH_NORM_MOMENTUM)
        self.apply_initial_bn = apply_initial_bn

        self.conv1 = nn.Conv2d(input_filters, filters*2, 1, 1, 0, bias=False)
        self.norm1 = nn.BatchNorm2d(filters*2, BATCH_NORM_MOMENTUM)

        self.atrous_conv = nn.Conv2d(filters*2, filters, 3, 1, dilation, dilation, bias=False)
        self.norm2 = nn.BatchNorm2d(filters, BATCH_NORM_MOMENTUM)

    def forward(self, input):
        if self.apply_initial_bn:
            input = self.initial_bn(input)

        input = self.conv1(input.relu())
        input = self.norm1(input)
        input = self.atrous_conv(input.relu())
        input = self.norm2(input)
        return input


class ASSPBlock(nn.Module):
    def __init__(self, input_filters=256, cat_filters=448, atrous_filters=128):
        super(ASSPBlock, self).__init__()

        self.atrous_conv_r3 = AtrousBlock(input_filters, atrous_filters, 3, apply_initial_bn=False)
        self.atrous_conv_r6 = AtrousBlock(cat_filters + atrous_filters, atrous_filters, 6)
        self.atrous_conv_r12 = AtrousBlock(cat_filters + atrous_filters*2, atrous_filters, 12)
        self.atrous_conv_r18 = AtrousBlock(cat_filters + atrous_filters*3, atrous_filters, 18)
        self.atrous_conv_r24 = AtrousBlock(cat_filters + atrous_filters*4, atrous_filters, 24)

        self.conv = nn.Conv2d(5 * atrous_filters + cat_filters, atrous_filters, 3, 1, 1, bias=ENABLE_BIAS)

    def forward(self, input):
        input, cat = input
        layer1_out = self.atrous_conv_r3(input)
        concat1 = torch.cat((cat, layer1_out), 1)

        layer2_out = self.atrous_conv_r6(concat1)
        concat2 = torch.cat((concat1, layer2_out), 1)

        layer3_out = self.atrous_conv_r12(concat2)
        concat3 = torch.cat((concat2, layer3_out), 1)

        layer4_out = self.atrous_conv_r18(concat3)
        concat4 = torch.cat((concat3, layer4_out), 1)

        layer5_out = self.atrous_conv_r24(concat4)
        concat5 = torch.cat((concat4, layer5_out), 1)

        features = activation_fn(self.conv(concat5))
        return features

# Code of this layer is taken from official PyTorch implementation
class LPGLayer(nn.Module):
    def __init__(self, scale):
        super(LPGLayer, self).__init__()
        self.scale = scale
        self.u = torch.arange(self.scale).reshape([1, 1, self.scale]).float()
        self.v = torch.arange(int(self.scale)).reshape([1, self.scale, 1]).float()

    def forward(self, plane_eq):
        plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.scale), 2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded, int(self.scale), 3)

        n1 = plane_eq_expanded[:, 0, :, :]
        n2 = plane_eq_expanded[:, 1, :, :]
        n3 = plane_eq_expanded[:, 2, :, :]
        n4 = plane_eq_expanded[:, 3, :, :]

        u = self.u.repeat(plane_eq.size(0), plane_eq.size(2) * int(self.scale), plane_eq.size(3)).to(device)
        u = (u - (self.scale - 1) * 0.5) / self.scale

        v = self.v.repeat(plane_eq.size(0), plane_eq.size(2), plane_eq.size(3) * int(self.scale)).to(device)
        v = (v - (self.scale - 1) * 0.5) / self.scale

        d = n4 / (n1 * u + n2 * v + n3)
        d = d.unsqueeze(1)
        return d


class Reduction(nn.Module):
    def __init__(self, scale, input_filters, is_final=False):
        super(Reduction, self).__init__()
        reduction_count = int(math.log(input_filters, 2)) - 2
        self.reductions = torch.nn.Sequential()
        for i in range(reduction_count):
            if i != reduction_count-1:
                self.reductions.add_module("1x1_reduc_%d_%d" % (scale, i), nn.Sequential(
                    nn.Conv2d(int(input_filters / math.pow(2, i)), int(input_filters / math.pow(2, i + 1)), 1, 1, 0, bias=ENABLE_BIAS),
                    activation_fn))
            else:
                if not is_final:
                    self.reductions.add_module("1x1_reduc_%d_%d" % (scale, i), nn.Sequential(
                        nn.Conv2d(int(input_filters / math.pow(2, i)), int(input_filters / math.pow(2, i + 1)), 1, 1, 0, bias=ENABLE_BIAS)))
                else:
                    self.reductions.add_module("1x1_reduc_%d_%d" % (scale, i), nn.Sequential(
                        nn.Conv2d(int(input_filters / math.pow(2, i)), 1, 1, 1, 0, bias=ENABLE_BIAS), nn.Sigmoid()))

    def forward(self, ip):
        return self.reductions(ip)


class LPGBlock(nn.Module):
    def __init__(self, scale, input_filters=128):
        super(LPGBlock, self).__init__()
        self.scale = scale

        self.reduction = Reduction(scale, input_filters)

        self.conv = nn.Conv2d(4, 3, 1, 1, 0)
        self.LPGLayer = LPGLayer(scale)

    def forward(self, input):
        input = self.reduction(input)

        plane_parameters = torch.zeros_like(input)
        input = self.conv(input)

        theta = input[:, 0, :, :].sigmoid() * 3.1415926535 / 6
        phi = input[:, 1, :, :].sigmoid() * 3.1415926535 * 2
        dist = input[:, 2, :, :].sigmoid() * MAX_DEPTH

        plane_parameters[:, 0, :, :] = torch.sin(theta) * torch.cos(phi)
        plane_parameters[:, 1, :, :] = torch.sin(theta) * torch.sin(phi)
        plane_parameters[:, 2, :, :] = torch.cos(theta)
        plane_parameters[:, 3, :, :] = dist

        plane_parameters[:, 0:3, :, :] = F.normalize(plane_parameters.clone()[:, 0:3, :, :], 2, 1)

        depth = self.LPGLayer(plane_parameters.float())
        return depth


class bts_encoder(nn.Module):
    def __init__(self):
        super(bts_encoder, self).__init__()
        self.dense_op_h2 = None
        self.dense_op_h4 = None
        self.dense_op_h8 = None
        self.dense_op_h16 = None
        self.dense_features = None

        self.dense_feature_extractor = self.initialize_dense_feature_extractor()
        self.freeze_batch_norm()
        self.initialize_hooks()

    def freeze_batch_norm(self):
        for module in self.dense_feature_extractor.modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.track_running_stats = True
                module.eval()
                module.affine = True
                module.requires_grad = True

    def initialize_dense_feature_extractor(self):
        dfe = torchvision.models.densenet161(True, True)
        dfe.features.denseblock1.requires_grad = False
        dfe.features.denseblock2.requires_grad = False
        dfe.features.conv0.requires_grad = False
        return dfe

    def set_h2(self, module, input_, output):
        self.dense_op_h2 = output

    def set_h4(self, module, input_, output):
        self.dense_op_h4 = output

    def set_h8(self, module, input_, output):
        self.dense_op_h8 = output

    def set_h16(self, module, input_, output):
        self.dense_op_h16 = output

    def set_dense_features(self, module, input_, output):
        self.dense_features = output

    def initialize_hooks(self):
        self.dense_feature_extractor.features.relu0.register_forward_hook(self.set_h2)
        self.dense_feature_extractor.features.pool0.register_forward_hook(self.set_h4)
        self.dense_feature_extractor.features.transition1.register_forward_hook(self.set_h8)
        self.dense_feature_extractor.features.transition2.register_forward_hook(self.set_h16)
        self.dense_feature_extractor.features.norm5.register_forward_hook(self.set_dense_features)

    def forward(self, ip):
        _ = self.dense_feature_extractor(ip)
        joint_input = (self.dense_features.relu(), self.dense_op_h2, self.dense_op_h4, self.dense_op_h8, self.dense_op_h16)
        return joint_input


class bts_decoder(nn.Module):
    def __init__(self):
        super(bts_decoder, self).__init__()
        self.UpscaleNet = UpscaleNetwork()
        self.DenseASSPNet = ASSPBlock()

        self.upscale_block3 = UpscaleBlock(64, 96, 128)  # H4
        self.upscale_block4 = UpscaleBlock(128, 96, 128)  # H2

        self.LPGBlock8 = LPGBlock(8, 128)
        self.LPGBlock4 = LPGBlock(4, 64)  # 64 Filter
        self.LPGBlock2 = LPGBlock(2, 64)  # 64 Filter

        self.upconv_h4 = UpscaleLayer(128, 64)
        self.upconv_h2 = UpscaleLayer(64, 32)  # 64 Filter
        self.upconv_h = UpscaleLayer(64, 32)  # 32 filter

        self.conv_h4 = nn.Conv2d(161, 64, 3, 1, 1, bias=ENABLE_BIAS)  # 64 Filter
        self.conv_h2 = nn.Conv2d(129, 64, 3, 1, 1, bias=ENABLE_BIAS)  # 64 Filter
        self.conv_h1 = nn.Conv2d(36, 32, 3, 1, 1, bias=ENABLE_BIAS)

        self.reduction1x1 = Reduction(1, 32, True)

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1, bias=ENABLE_BIAS)

    def forward(self, joint_input, focal):
        (dense_features, dense_op_h2, dense_op_h4, dense_op_h8, dense_op_h16) = joint_input
        upscaled_out = self.UpscaleNet(joint_input)

        dense_assp_out = self.DenseASSPNet(upscaled_out)

        upconv_h4 = self.upconv_h4(dense_assp_out)
        depth_8x8 = self.LPGBlock8(dense_assp_out) / MAX_DEPTH
        depth_8x8_ds = nn.functional.interpolate(depth_8x8, scale_factor=1 / 4, mode="nearest")
        depth_concat_4x4 = torch.cat((depth_8x8_ds, dense_op_h4, upconv_h4), 1)

        conv_h4 = activation_fn(self.conv_h4(depth_concat_4x4))
        upconv_h2 = self.upconv_h2(conv_h4)
        depth_4x4 = self.LPGBlock4(conv_h4) / MAX_DEPTH

        depth_4x4_ds = nn.functional.interpolate(depth_4x4, scale_factor=1 / 2, mode="nearest")
        depth_concat_2x2 = torch.cat((depth_4x4_ds, dense_op_h2, upconv_h2), 1)

        conv_h2 = activation_fn(self.conv_h2(depth_concat_2x2))
        upconv_h = self.upconv_h(conv_h2)
        depth_1x1 = self.reduction1x1(upconv_h)
        depth_2x2 = self.LPGBlock2(conv_h2) / MAX_DEPTH

        depth_concat = torch.cat((upconv_h, depth_1x1, depth_2x2, depth_4x4, depth_8x8), 1)
        depth = activation_fn(self.conv_h1(depth_concat))
        depth = self.final_conv(depth).sigmoid() * MAX_DEPTH + DEPTH_OFFSET

        depth *= focal.view(-1, 1, 1, 1) / 715.0873
        return depth, depth_2x2, depth_4x4, depth_8x8


class bts_model(nn.Module):
    def __init__(self):
        super(bts_model, self).__init__()
        self.encoder = bts_encoder()
        self.decoder = bts_decoder()

    def forward(self, input, focal=715.0873):
        joint_input = self.encoder(input)
        return self.decoder(joint_input, focal)


class SilogLoss(nn.Module):
    def __init__(self):
        super(SilogLoss, self).__init__()

    def forward(self, ip, target, ratio=10, ratio2=0.85):
        ip = ip.reshape(-1)
        target = target.reshape(-1)

        mask = (target > 1) & (target < 81)
        masked_ip = torch.masked_select(ip.float(), mask)
        masked_op = torch.masked_select(target, mask)

        log_diff = torch.log(masked_ip * ratio) - torch.log(masked_op * ratio)
        log_diff_masked = log_diff

        silog1 = torch.mean(log_diff_masked ** 2)
        silog2 = ratio2 * (torch.mean(log_diff_masked) ** 2)
        silog_loss = torch.sqrt(silog1 - silog2) * ratio
        return silog_loss


class BtsController:
    def __init__(self, log_directory='run_1', logs_folder='tensorboard', backprop_frequency=1):
        self.bts = bts_model().float().to(device)
        self.optimizer = torch.optim.AdamW([{'params': self.bts.encoder.parameters(), 'weight_decay': 1e-2},
                                       {'params': self.bts.decoder.parameters(), 'weight_decay': 0}],
                                      lr=1e-4, eps=1e-6)

        if USE_APEX:
            self.bts, self.optimizer = apex.amp.initialize(self.bts, self.optimizer, opt_level=APEX_OPT_LEVEL)

        self.bts = torch.nn.DataParallel(self.bts)

        self.backprop_frequency = backprop_frequency

        log_path = os.path.join(logs_folder, log_directory)
        self.writer = SummaryWriter(log_path)

        self.criterion = SilogLoss()

        self.learning_rate_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)

        self.current_epoch = 0
        self.last_loss = 0
        self.current_step = 0

    def eval(self):
        self.bts = self.bts.eval()

    def train(self):
        self.bts = self.bts.train()

    def predict(self, input, is_channels_first=True, focal=715.0873, normalize=False):
        if normalize:
            input = A.Compose([A.Normalize()])(**{"image": input})["image"]

        if is_channels_first:
            tensor_input = torch.tensor(input).unsqueeze(-1).to(device).float().transpose(0, 3).transpose(2,
                                                                                                          3).transpose(
                1, 2)
        else:
            tensor_input = torch.tensor(input).unsqueeze(-1).to(device).float().transpose(0, 3).transpose(1,
                                                                                                          2).transpose(
                2, 3)

        shape_changed = False
        org_shape = tensor_input.shape[2:]
        if org_shape[0] % 32 != 0 or org_shape[1] % 32 != 0:
            shape_changed = True
            new_shape_y = round(org_shape[0] / 32) * 32
            new_shape_x = round(org_shape[1] / 32) * 32
            tensor_input = F.interpolate(tensor_input, (new_shape_y, new_shape_x), mode="bilinear")

        model_output = self.bts(tensor_input, torch.tensor(focal).unsqueeze(0))[0][0].detach().unsqueeze(0)
        if shape_changed:
            model_output = F.interpolate(model_output, (org_shape[0], org_shape[1]), mode="nearest")

        return model_output.cpu().squeeze()

    @staticmethod
    def depth_map_to_rgbimg(depth_map):
        depth_map = np.asarray(np.squeeze((255 - torch.clamp_max(depth_map * 4, 250)).byte().numpy()), np.uint8)
        depth_map = np.asarray(cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB), np.uint8)
        return depth_map

    @staticmethod
    def normalize_img(image):
        transformation = A.Compose([
            A.Normalize()
        ])
        return transformation(**{"image": image})["image"]

    def run_train_step(self, tensor_input, tensor_output, tensor_focal):
        tensor_input, tensor_output = tensor_input.to(device), tensor_output.to(device)
        # Get Models prediction and calculate loss
        model_output, depth2, depth4, depth8 = self.bts(tensor_input, tensor_focal)

        loss = self.criterion(model_output, tensor_output) * 1/self.backprop_frequency

        if USE_APEX:
            with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if self.current_step % self.backprop_frequency == 0:  # Make update once every x steps
            torch.nn.utils.clip_grad_norm_(self.bts.parameters(), 5)
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.current_step % 100 == 0:
            self.writer.add_scalar("Loss", loss.item() * self.backprop_frequency / tensor_input.shape[0], self.current_step)

        if self.current_step % 1000 == 0:
            img = tensor_input[0].detach().transpose(0, 2).transpose(0, 1).cpu().numpy().astype(np.uint8)
            self.writer.add_image("Input", img, self.current_step, None, "HWC")

            visual_result = (255-torch.clamp_max(torchvision.utils.make_grid([tensor_output[0], model_output[0]]) * 5, 250)).byte()

            self.writer.add_image("Output/Prediction", visual_result, self.current_step)
            depths = [depth2[0], depth4[0], depth8[0]]
            depths = [depth*MAX_DEPTH for depth in depths]
            depth_visual = (255-torch.clamp_max(torchvision.utils.make_grid(depths) * 5, 250)).byte()

            self.writer.add_image("Depths", depth_visual, self.current_step)

        self.current_step += 1

    def save_model(self, path):
        save_dict = {
            'epoch': self.current_epoch,
            'model_state_dict': self.bts.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            "scheduler_state_dict": self.learning_rate_scheduler.state_dict(),
            'loss': self.last_loss,
            "last_step": self.current_step
        }
        if USE_APEX:
            save_dict["amp"] = apex.amp.state_dict()
            save_dict["opt_level"] = APEX_OPT_LEVEL

        torch.save(save_dict, path)

    def load_model(self, path):
        dict = torch.load(path)

        if USE_APEX:
            saved_opt_level = dict["opt_level"]
            self.bts, self.optimizer = apex.amp.initialize(self.bts, self.optimizer, opt_level=saved_opt_level)
            apex.amp.load_state_dict(dict["amp"])

        self.current_epoch = dict["epoch"]
        self.bts.load_state_dict(dict["model_state_dict"])
        self.bts = self.bts.float().to(device)

        self.optimizer.load_state_dict(dict["optimizer_state_dict"])

        self.learning_rate_scheduler.load_state_dict(dict["scheduler_state_dict"])
        self.last_loss = dict["loss"]
        self.current_step = dict["last_step"]

        return dict