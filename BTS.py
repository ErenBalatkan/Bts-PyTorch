import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torch import optim
import os
import math
from numba import cuda
import cv2
import albumentations as A

from torch.utils.tensorboard import SummaryWriter

activation_fn = nn.ELU()

MAX_DEPTH = 83
DEPTH_OFFSET = 0.01 # This is used for ensuring depth prediction gets into positive range


class UpscaleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpscaleLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        input = nn.functional.interpolate(input, scale_factor=2)
        input = activation_fn(self.conv(input))
        input = self.bn(input)
        return input


class UpscaleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpscaleBlock, self).__init__()
        self.uplayer = UpscaleLayer(in_channels, out_channels)
        self.conv = nn.Conv2d(out_channels+skip_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

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

        self.initial_bn = nn.BatchNorm2d(input_filters)
        self.apply_initial_bn = apply_initial_bn

        self.conv1 = nn.Conv2d(input_filters, filters*2, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(filters*2)

        self.atrous_conv = nn.Conv2d(filters*2, filters, 3, 1, dilation, dilation)
        self.norm2 = nn.BatchNorm2d(filters)

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

        self.conv = nn.Conv2d(5 * atrous_filters + cat_filters, atrous_filters, 3, 1, 1)

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


@cuda.jit
def Cuda_LPG_Forward(plane_parameters, scale, result):
    batch_idx = cuda.threadIdx.z
    pixel_y = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pixel_x = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if pixel_x < plane_parameters.shape[3] and pixel_y < plane_parameters.shape[2]:
        n1 = plane_parameters[batch_idx, 0, pixel_y, pixel_x]
        n2 = plane_parameters[batch_idx, 1, pixel_y, pixel_x]
        n3 = plane_parameters[batch_idx, 2, pixel_y, pixel_x]
        n4 = plane_parameters[batch_idx, 3, pixel_y, pixel_x]

        for ii in range(scale):
            for jj in range(scale):

                v = (ii - (scale-1) / 2.0) / scale / 715.0
                u = (jj - (scale-1) / 2.0) / scale / 715.0

                numerator = n4
                denominator = (n1*u + n2*v + n3)

                result[batch_idx, 0, pixel_y*scale + ii, pixel_x*scale + jj] = numerator/denominator



@cuda.jit
def Cuda_LPG_Backward(plane_parameters, scale, grad_outputs, grad_plane_parameters):
    batch_idx = cuda.threadIdx.z
    pixel_y = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pixel_x = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if pixel_x < plane_parameters.shape[3] and pixel_y < plane_parameters.shape[2]:

        # if(grad_outputs[batch_idx, 0, pixel_y, pixel_x] == 0):
        #     print(grad_outputs[batch_idx, 0, pixel_y, pixel_x])

        n1 = plane_parameters[batch_idx, 0, pixel_y, pixel_x]
        n2 = plane_parameters[batch_idx, 1, pixel_y, pixel_x]
        n3 = plane_parameters[batch_idx, 2, pixel_y, pixel_x]
        n4 = plane_parameters[batch_idx, 3, pixel_y, pixel_x]

        for ii in range(scale):
            for jj in range(scale):
                v = (ii - (scale-1) / 2.0) / scale / 715.0
                u = (jj - (scale-1) / 2.0) / scale / 715.0

                denominator = n1 * u + n2 * v + n3
                denominator_sq = denominator ** 2

                grad_plane_parameters[batch_idx, 0, pixel_y, pixel_x] += grad_outputs[batch_idx, 0, pixel_y, pixel_x] * (-1.0 * u) / denominator_sq
                grad_plane_parameters[batch_idx, 1, pixel_y, pixel_x] += grad_outputs[batch_idx, 0, pixel_y, pixel_x] * (-1.0 * v) / denominator_sq
                grad_plane_parameters[batch_idx, 2, pixel_y, pixel_x] += grad_outputs[batch_idx, 0, pixel_y, pixel_x] * (-1.0) / denominator_sq
                grad_plane_parameters[batch_idx, 3, pixel_y, pixel_x] += grad_outputs[batch_idx, 0, pixel_y, pixel_x] / denominator



class LPGLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, plane_parameters, scale):
        ctx.save_for_backward(plane_parameters, torch.tensor((scale)))

        plane_shape = list(plane_parameters.shape)
        scaled_shape = plane_shape.copy()
        scaled_shape[3] *= scale
        scaled_shape[2] *= scale
        scaled_shape[1] = 1

        depth = np.ones(shape=scaled_shape)

        gpu_plane_parameters = cuda.to_device(plane_parameters.cpu().numpy())
        gpu_results = cuda.to_device(depth)

        threads = [20, 20, plane_shape[0]]
        blocks = [math.ceil(plane_parameters.shape[2] / 20), math.ceil(plane_parameters.shape[3] / 20)]

        Cuda_LPG_Forward[threads, blocks](gpu_plane_parameters, scale, gpu_results)
        gpu_results.copy_to_host(depth)

        cuda.synchronize()
        depth = torch.tensor(depth).float().cuda()

        return depth

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_plane_parameters = grad_scale = None
        plane_parameters, scale = ctx.saved_tensors

        grad_plane_parameters = np.zeros_like(plane_parameters.cpu())

        scale = scale.item()

        plane_shape = list(plane_parameters.shape)

        gpu_plane_parameters = cuda.to_device(plane_parameters.cpu().numpy())
        gpu_grad_output = cuda.to_device(grad_outputs.cpu().numpy())

        gpu_grad_plane_parameters = cuda.to_device(grad_plane_parameters)

        threads = [20, 20, plane_shape[0]]
        blocks = [math.ceil(plane_parameters.shape[2] / 20), math.ceil(plane_parameters.shape[3] / 20)]

        Cuda_LPG_Backward[threads, blocks](gpu_plane_parameters, scale, gpu_grad_output, gpu_grad_plane_parameters)
        gpu_grad_plane_parameters.copy_to_host(grad_plane_parameters)

        cuda.synchronize()

        grad_plane_parameters = torch.tensor(grad_plane_parameters).float().cuda()

        return grad_plane_parameters, grad_scale


class LPGBlock(nn.Module):
    def __init__(self, scale, input_filters=128):
        super(LPGBlock, self).__init__()
        self.scale = scale

        reduction_count = int(math.log(input_filters, 2)) - 2
        self.reductions = []
        for i in range(reduction_count):
            self.reductions.append(nn.Conv2d(int(input_filters / math.pow(2, i)), int(input_filters / math.pow(2, i+1)),
                                             1, 1, 0))

        self.reductions = [x.cuda() for x in self.reductions]

        self.conv = nn.Conv2d(4, 3, 1, 1, 0)

        self.LPGLayer = LPGLayer.apply

    def forward(self, input):
        for reduction_idx in range(len(self.reductions)):
            reduction = self.reductions[reduction_idx]
            input = activation_fn(reduction(input))

        plane_parameters = torch.zeros_like(input)
        input = self.conv(input)

        theta = input[:, 0, :, :].sigmoid() * 3.1415926535 / 6
        phi = input[:, 1, :, :].sigmoid() * 3.1415926535 * 2
        dist = input[:, 2, :, :].sigmoid() * MAX_DEPTH

        plane_parameters[:, 0, :, :] = torch.sin(theta) * torch.cos(phi)
        plane_parameters[:, 1, :, :] = torch.sin(theta) * torch.sin(phi)
        plane_parameters[:, 2, :, :] = torch.cos(theta)
        plane_parameters[:, 3, :, :] = dist

        # plane_parameters[:, 0, :, :] = plane_parameters[:, 0, :, :].tanh()
        # plane_parameters[:, 1, :, :] = plane_parameters[:, 0, :, :].tanh()
        # plane_parameters[:, 2, :, :] = plane_parameters[:, 0, :, :].sigmoid()
        # plane_parameters[:, 3, :, :] = plane_parameters[:, 0, :, :].sigmoid() * MAX_DEPTH

        plane_parameters[:, 0:3, :, :] = F.normalize(plane_parameters.clone()[:, 0:3, :, :], 2, 1)

        depth = self.LPGLayer(plane_parameters, self.scale)

        return depth


class bts_eren(nn.Module):
    def __init__(self):
        super(bts_eren, self).__init__()

        self.dense_op_h2 = None
        self.dense_op_h4 = None
        self.dense_op_h8 = None
        self.dense_op_h16 = None
        self.dense_features = None

        self.dense_feature_extractor = self.initialize_dense_feature_extractor()
        self.freeze_batch_norm()
        self.initialize_hooks()

        self.UpsccaleNet = UpscaleNetwork()
        self.DenseASSPNet = ASSPBlock()

        self.upscale_block3 = UpscaleBlock(64, 96, 128)  # H4
        self.upscale_block4 = UpscaleBlock(128, 96, 128)  # H2

        self.LPGBlock8 = LPGBlock(8, 128)
        self.LPGBlock4 = LPGBlock(4, 64) # 64 Filter
        self.LPGBlock2 = LPGBlock(2, 64) # 64 Filter

        self.upconv_h4 = UpscaleLayer(128, 64)
        self.upconv_h2 = UpscaleLayer(64, 32) # 64 Filter
        self.upconv_h = UpscaleLayer(64, 32) # 32 filter

        self.conv_h4 = nn.Conv2d(161, 64, 3, 1, 1) # 64 Filter
        self.conv_h2 = nn.Conv2d(129, 64, 3, 1, 1) # 64 Filter
        self.conv_h1 = nn.Conv2d(35, 32, 3, 1, 1)

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def freeze_batch_norm(self):
        for module in self.dense_feature_extractor.modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm3d):
                module.eval()

    def initialize_dense_feature_extractor(self):
        dfe = torchvision.models.densenet161(True, True)
        dfe.features.denseblock1.requires_grad = False
        dfe.features.denseblock2.requires_grad = False
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
        self.dense_feature_extractor.features.transition1.pool.register_forward_hook(self.set_h8)
        self.dense_feature_extractor.features.transition2.pool.register_forward_hook(self.set_h16)
        self.dense_feature_extractor.features.norm5.register_forward_hook(self.set_dense_features)

    def forward(self, input):
        _ = self.dense_feature_extractor(input)
        joint_input = (self.dense_features, self.dense_op_h2, self.dense_op_h4, self.dense_op_h8, self.dense_op_h16)
        upscaled_out = self.UpsccaleNet(joint_input)
        dense_assp_out = self.DenseASSPNet(upscaled_out)

        upconv_h4 = self.upconv_h4(dense_assp_out)
        depth_8x8 = self.LPGBlock8(dense_assp_out) / MAX_DEPTH
        depth_8x8_ds = nn.functional.interpolate(depth_8x8, scale_factor=1/4)
        depth_concat_4x4 = torch.cat((depth_8x8_ds, self.dense_op_h4, upconv_h4), 1)

        conv_h4 = activation_fn(self.conv_h4(depth_concat_4x4))
        upconv_h2 = self.upconv_h2(conv_h4)
        depth_4x4 = self.LPGBlock4(conv_h4) / MAX_DEPTH

        depth_4x4_ds = nn.functional.interpolate(depth_4x4, scale_factor=1/2)
        depth_concat_2x2 = torch.cat((depth_4x4_ds, self.dense_op_h2, upconv_h2), 1)

        conv_h2 = activation_fn(self.conv_h2(depth_concat_2x2))
        upconv_h = self.upconv_h(conv_h2)
        depth_2x2 = self.LPGBlock2(conv_h2) / MAX_DEPTH
        depth_concat = torch.cat((depth_2x2, upconv_h, depth_4x4, depth_8x8), 1)

        depth = activation_fn(self.conv_h1(depth_concat))

        depth = self.final_conv(depth).sigmoid() * MAX_DEPTH + DEPTH_OFFSET

        return depth, depth_2x2, depth_4x4, depth_8x8


class SilogLoss(nn.Module):
    def __init__(self):
        super(SilogLoss, self).__init__()

    def forward(self, ip, target, ratio=10, ratio2=0.85):
        ip = ip.reshape(-1)
        target = target.reshape(-1)

        mask = (target > 0.01) & (target < 81)
        masked_ip = torch.masked_select(ip, mask)
        masked_op = torch.masked_select(target, mask)

        log_diff = torch.log(masked_ip * ratio) - torch.log(masked_op * ratio)
        log_diff_masked = log_diff

        silog1 = torch.mean(log_diff_masked ** 2)
        silog2 = ratio2 * (torch.mean(log_diff_masked) ** 2)
        silog_loss = torch.sqrt(silog1 - silog2) * ratio
        return silog_loss


class BtsController:
    def __init__(self, log_directory='run_1', logs_folder='tensorboard', backprop_frequency=8):
        self.bts = bts_eren().float().cuda()

        self.backprop_frequency = backprop_frequency

        log_path = os.path.join(logs_folder, log_directory)
        self.writer = SummaryWriter(log_path)

        self.criterion = SilogLoss()
        self.optimizer = optim.Adam(self.bts.parameters(), 1e-4)
        self.learning_rate_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.97)

        self.current_epoch = 0
        self.last_loss = 0
        self.current_step = 0

    def predict(self, input, is_channels_first=True):
        if is_channels_first:
            tensor_input = torch.tensor(input).unsqueeze(-1).cuda().float().transpose(0, 3).transpose(2, 3).transpose(1, 2)
        else:
            tensor_input = torch.tensor(input).unsqueeze(-1).cuda().float().transpose(0, 3).transpose(1, 2).transpose(2, 3)

        model_output = self.bts(tensor_input)[0][0].detach().cpu().transpose(0, 1).transpose(1, 2).squeeze(-1)
        return model_output

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

    def run_train_step(self, tensor_input, tensor_output):
        tensor_input, tensor_output = tensor_input.cuda(), tensor_output.cuda()
        # Get Models prediction and calculate loss
        model_output, depth2, depth4, depth8 = self.bts(tensor_input)

        # print("expecteed output:", tensor_output.shape)
        # print("model_output:", model_output.shape)
        loss = self.criterion(model_output, tensor_output) * 1/self.backprop_frequency
        loss.backward()
        # loss = torch.nn.functional.mse_loss(model_output, tensor_output)

        if self.current_step % self.backprop_frequency == 0:  # Make update once every x steps
            torch.nn.utils.clip_grad_norm_(self.bts.parameters(), 5)
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.current_step % 10 == 0:
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
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.bts.state_dict(),
            "dfe_state_dict": self.bts.dense_feature_extractor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            "scheduler_state_dict": self.learning_rate_scheduler.state_dict(),
            'loss': self.last_loss,
            "last_step": self.current_step
        }, path)

    def load_model(self, path):
        dict = torch.load(path)

        self.current_epoch = dict["epoch"]
        self.bts.load_state_dict(dict["model_state_dict"])
        self.bts.dense_feature_extractor.load_state_dict(dict["dfe_state_dict"])
        self.bts = self.bts.float().cuda()

        self.optimizer.load_state_dict(dict["optimizer_state_dict"])
        self.learning_rate_scheduler.load_state_dict(dict["scheduler_state_dict"])
        self.last_loss = dict["loss"]
        self.current_step = dict["last_step"]

        self.bts.freeze_batch_norm()

        return dict