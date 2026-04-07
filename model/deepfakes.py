import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import os
import argparse
from PIL import Image
import cv2
from project_utils import norm, denorm

""" For the deepfake model utilities, comment out the ones that are not used. 
    Please manage to download the original model code and weights.
"""

# SimSwap utilities.
from SimSwap.models.models import create_model
from SimSwap.options.test_options import TestOptions


class SimSwapModel(nn.Module):

    def __init__(self, img_size):
        super(SimSwapModel, self).__init__()
        opt = TestOptions().parse()
        self.img_size = img_size
        if self.img_size == 128:
            opt.crop_size = 224
            opt.image_size = 224
            opt.netG = 'global'
        else:
            opt.crop_size = 512
            opt.image_size = 512
            opt.netG = '550000'
        self.sim_swap = create_model(opt)
        self.sim_swap.eval()
        self.arcface_norm = transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225]
        )

    def one_step_swap(self, source, target, device):
        """ Single directional face-swap from source to target. """
        source = self.arcface_norm(source)
        # (112, 112) required by SimSwap.
        source_downsample = F.interpolate(source, size=(112, 112))
        latent_source = self.sim_swap.netArc(source_downsample)
        latent_source = latent_source.detach().to('cpu')
        latent_source = latent_source / np.linalg.norm(latent_source, axis=1, keepdims=True)
        latent_source = latent_source.to(device)

        # The swapped face has facial id of source but attributes of target.
        swapped_face = self.sim_swap(source, target, latent_source, latent_source, True)
        return swapped_face

    def forward(self, img_wtm_device):
        """ When trained on a batch along with ID-Mark, faces are to be swapped within the batch.
            Source face at index [i + 1] is swapped onto the target face at index [i].
        """
        img_wtm = img_wtm_device[0]
        img = img_wtm_device[1]
        device = img_wtm_device[2]

        if self.img_size == 128:
            resize = transforms.Resize((224, 224))
        else:
            resize = transforms.Resize((512, 512))
        resize_back = transforms.Resize((self.img_size, self.img_size))
        img_wtm = resize(img_wtm)
        img = resize(img)
        img_source = torch.roll(img, 1, 0)
        swapped_face_wtm = self.one_step_swap(denorm(img_source), denorm(img_wtm), device)
        swapped_face_wtm = norm(resize_back(swapped_face_wtm))
        return swapped_face_wtm


# InfoSwap utilities.
from infoswap.infoswap.modules.encoder128 import Backbone128
from infoswap.infoswap.modules.iib import IIB
from infoswap.infoswap.modules.aii_generator import AII512
from infoswap.infoswap.modules.decoder512 import UnetDecoder512
from infoswap.infoswap.preprocess.mtcnn import MTCNN


class InfoSwapModel(nn.Module):
    def __init__(self, device):
        super(InfoSwapModel, self).__init__()
        self.mtcnn = MTCNN()
        self.transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.device = device

        """ Prepare Models: """
        root = 'infoSwap/checkpoints_512/w_kernel_smooth'

        pathG = 'ckpt_ks_G.pth'
        pathE = 'ckpt_ks_E.pth'
        pathI = 'ckpt_ks_I.pth'

        self.encoder = Backbone128(50, 0.6, 'ir_se').eval().to(self.device)
        state_dict = torch.load('infoSwap/modules/model_128_ir_se50.pth', map_location=self.device)
        self.encoder.load_state_dict(state_dict, strict=True)
        self.G = AII512().eval().to(self.device)
        self.decoder = UnetDecoder512().eval().to(self.device)

        # Define Information Bottlenecks:
        self.N = 10
        _ = self.encoder(torch.rand(1, 3, 128, 128).to(device), cache_feats=True)
        _readout_feats = self.encoder.features[:(self.N + 1)]  # one layer deeper than the z_attrs needed
        in_c = sum(map(lambda f: f.shape[-3], _readout_feats))
        out_c_list = [_readout_feats[i].shape[-3] for i in range(self.N)]

        self.iib = IIB(in_c, out_c_list, device, smooth=True, kernel_size=1)
        self.iib = self.iib.eval()
        
        self.G.load_state_dict(torch.load(os.path.join(root, pathG), map_location=self.device), strict=True)
        print("Successfully load G!")
        self.decoder.load_state_dict(torch.load(os.path.join(root, pathE), map_location=self.device), strict=True)
        print("Successfully load Decoder!")
        # 3) load IIB:
        self.iib.load_state_dict(torch.load(os.path.join(root, pathI), map_location=self.device), strict=True)
        print("Successfully load IIB!")

        self.param_dict = []
        for i in range(self.N + 1):
            state = torch.load(f'infoSwap/modules/weights128/readout_layer{i}.pth', map_location=self.device)
            n_samples = state['n_samples'].float()
            std = torch.sqrt(state['s'] / (n_samples - 1)).to(self.device)
            neuron_nonzero = state['neuron_nonzero'].float()
            active_neurons = (neuron_nonzero / n_samples) > 0.01
            self.param_dict.append([state['m'].to(self.device), std, active_neurons])

    def one_step_swap(self, source_path, target_path):
        if target_path.endswith('.png') or target_path.endswith('.jpg'):
            tar_list = [target_path, ]
        else:
            tmp_list = [f for f in os.listdir(target_path) if f.endswith('jpg') or f.endswith('png')]
            tar_list = sorted(tmp_list)
        M = len(tar_list)
        Xs = cv2.imread(source_path)
        Xs = Image.fromarray(Xs)
        Xs = self.transform(Xs).unsqueeze(0)
        Xs = Xs.to(self.device)

        for idx in range(M):
            tar_img_path = target_path
            print(tar_img_path)
            prefix = tar_list[idx].split('.')[0]
            suffix = tar_img_path.split('.')[-1]
            save_path = os.path.join('/temp_data/infoswap_images', prefix, '_gen.', suffix)
            if os.path.exists(save_path):
                continue

            with torch.no_grad():
                '''(1) load Xt: '''
                print(target_path, end=', ')
                xt = cv2.imread(tar_img_path)
                print(xt.shape)

                Xt = Image.fromarray(xt)

                '''(2) generate Y: '''
                B = 1
                Xt = self.transform(Xt).unsqueeze(0).to(self.device)
               
                X_id = self.encoder(
                    F.interpolate(torch.cat((Xs, Xt), dim=0)[:, :, 37:475, 37:475], size=[128, 128],
                                mode='bilinear', align_corners=True),
                    cache_feats=True
                )
                # 01 Get Inter-features After One Feed-Forward:
                # batch size is 2 * B, [:B] for Xs and [B:] for Xt
                min_std = torch.tensor(0.01, device=self.device)
                readout_feats = [(self.encoder.features[i] - self.param_dict[i][0]) / torch.max(self.param_dict[i][1], min_std)
                                for i in range(self.N + 1)]

                # 02 information restriction:
                X_id_restrict = torch.zeros_like(X_id).to(self.device)  # [2*B, 512]
                Xt_feats, X_lambda = [], []
                Xt_lambda = []
                Rs_params, Rt_params = [], []
                for i in range(self.N):
                    R = self.encoder.features[i]  # [2*B, Cr, Hr, Wr]
                    Z, lambda_, _ = getattr(self.iib, f'iba_{i}')(
                        R, readout_feats,
                        m_r=self.param_dict[i][0], std_r=self.param_dict[i][1],
                        active_neurons=self.param_dict[i][2],
                    )
                    X_id_restrict += self.encoder.restrict_forward(Z, i)

                    Rs, Rt = R[:B], R[B:]
                    lambda_s, lambda_t = lambda_[:B], lambda_[B:]

                    m_s = torch.mean(Rs, dim=0)  # [C, H, W]
                    std_s = torch.mean(Rs, dim=0)
                    Rs_params.append([m_s, std_s])

                    eps_s = torch.randn(size=Rt.shape).to(Rt.device) * std_s + m_s
                    feat_t = Rt * (1. - lambda_t) + lambda_t * eps_s

                    Xt_feats.append(feat_t)  # only related with lambda
                    Xt_lambda.append(lambda_t)

                X_id_restrict /= float(self.N)
                Xs_id = X_id_restrict[:B]
                Xt_feats[0] = Xt
                Xt_attr, Xt_attr_lamb = self.decoder(Xt_feats, lambs=Xt_lambda, use_lambda=True)

                Y = self.G(Xs_id, Xt_attr, Xt_attr_lamb)
            return Y

    def forward(self, source_path, target_wm_path):
        swapped_wm_face = self.one_step_swap(source_path, target_wm_path)
        if self.img_size == 256:
            resize_back = transforms.Resize((256, 256))
        else:
            resize_back = transforms.Resize((512, 512))
        swapped_wm_face = resize_back(swapped_wm_face)
        return swapped_wm_face


# UniFace utilities.
from UniFace.generate_swap import Model as UniFace


class UniFaceModel(nn.Module):
    def __init__(self, device):
        super(UniFaceModel, self).__init__()
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--mixing_type",
            type=str,
            default='examples'
        )
        parser.add_argument("--inter", type=str, default='pair')
        parser.add_argument("--ckpt", type=str, default='session/swap/checkpoints/500000.pt')
        parser.add_argument("--test_path", type=str, default='examples/img/')
        parser.add_argument("--test_txt_path", type=str, default='examples/pair_swap.txt')
        parser.add_argument("--batch", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--save_image_dir", type=str, default="expr")

        args = parser.parse_args()

        ckpt = torch.load(args.ckpt, weights_only=False)
        train_args = ckpt["train_args"]
        for key in vars(train_args):
            if not (key in vars(args)):
                setattr(args, key, getattr(train_args, key))
        self.swap_model = UniFace(args).half().to(device)
        self.swap_model.g_ema.load_state_dict(ckpt["g_ema"])
        self.swap_model.e_ema.load_state_dict(ckpt["e_ema"])
        self.swap_model.eval()

    def forward(self, img_wtm_device):
        img_wtm = img_wtm_device[0]
        img = img_wtm_device[1]

        img_size = img_wtm.shape[-1]
        resize_back = transforms.Resize((img_size, img_size))

        img_source = torch.roll(img, 5, 0)
        _, _, swapped_img_wtm = self.swap_model([img_wtm.half(), img_source.half()])
        swapped_face_wtm = resize_back(swapped_img_wtm)
        return swapped_face_wtm.type(torch.cuda.FloatTensor)


# E4S utilities.
from e4s.src.pretrained.face_vid2vid.driven_demo import init_facevid2vid_pretrained_model, drive_source_demo
from e4s.src.pretrained.gpen.gpen_demo import init_gpen_pretrained_model, GPEN_demo
from e4s.src.pretrained.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo, \
    vis_parsing_maps
from e4s.src.utils.swap_face_mask import swap_head_mask_revisit_considerGlass
from e4s.src.utils import torch_utils as e4s_utils
from e4s.src.utils.alignmengt import crop_faces, calc_alignment_coefficients
from e4s.src.utils.morphology import dilation, erosion
from e4s.src.utils.multi_band_blending import blending
from e4s.src.options.swap_options import SwapFacePipelineOptions as E4SSwapOptions
from e4s.src.models.networks import Net3
from e4s.src.datasets.dataset import TO_TENSOR, NORMALIZE, __celebAHQ_masks_to_faceParser_mask_detailed
from e4s.scripts.face_swap import create_masks, logical_or_reduce, logical_and_reduce, paste_image_mask, paste_image, \
    smooth_face_boundry, crop_and_align_face, swap_comp_style_vector
from skimage.transform import resize


class E4SModel(nn.Module):

    def __init__(self,device):
        super(E4SModel, self).__init__()
        self.opts = E4SSwapOptions().parse()
        self.generator, self.kp_detector, self.he_estimator, self.estimate_jacobian = self.init_facevid2vid_pretrained_model()
        self.GPEN_model = self.init_gpen_pretrained_model()
        self.faceParsing_model = self.init_faceParsing_pretrained_model()
        self.net = self.init_e4s_pretrained_model()
        self.device = device

    def init_facevid2vid_pretrained_model(self):
        face_vid2vid_cfg = "e4s/pretrained_ckpts/facevid2vid/vox-256.yaml"
        face_vid2vid_ckpt = "e4s/pretrained_ckpts/facevid2vid/00000189-checkpoint.pth.tar"
        generator, kp_detector, he_estimator, estimate_jacobian = init_facevid2vid_pretrained_model(
            face_vid2vid_cfg,
            face_vid2vid_ckpt
        )
        return generator, kp_detector, he_estimator, estimate_jacobian

    def init_gpen_pretrained_model(self):
        gpen_model_params = {
            "base_dir": "e4s/pretrained_ckpts/gpen/",
            "in_size": 512,
            "model": "GPEN-BFR-512",
            "use_sr": True,
            "sr_model": "realesrnet",
            "sr_scale": 4,
            "channel_multiplier": 2,
            "narrow": 1,
        }
        GPEN_model = init_gpen_pretrained_model(model_params=gpen_model_params)
        return GPEN_model

    def init_faceParsing_pretrained_model(self):
        faceParser_ckpt = "e4s/pretrained_ckpts/face_parsing/79999_iter.pth"
        faceParsing_model = init_faceParsing_pretrained_model(self.opts.faceParser_name, faceParser_ckpt, "")
        print('Loaded pre-trained face parsing models successfully.')
        return faceParsing_model

    def init_e4s_pretrained_model(self):
        net = Net3(self.opts)
        net = net.to(self.opts.device)
        save_dict = torch.load(self.opts.checkpoint_path)
        net.load_state_dict(e4s_utils.remove_module_prefix(save_dict["state_dict"], prefix="module."))
        net.latent_avg = save_dict['latent_avg'].to(self.opts.device)
        print("Loaded E4S pre-trained model successfully!")
        return net

    def faceSwapping_pipeline(self, source, target, save_dir, target_mask=None, only_target_crop=False, need_crop=False,
                              verbose=False):

        os.makedirs(save_dir, exist_ok=True)
        result = []
        for i in range(len(source)):
            source_image = source[i].cpu().detach().numpy()
            target_image = target[i].cpu().detach().numpy()

            source_array = np.clip(source_image * 255, 0, 255).astype(np.uint8)
            target_array = np.clip(target_image * 255, 0, 255).astype(np.uint8)

            source_array = source_array.transpose((1, 2, 0))
            target_array = target_array.transpose((1, 2, 0))

            S = Image.fromarray(source_array).convert('RGB').resize((1024, 1024))
            T = Image.fromarray(target_array).convert('RGB').resize((1024, 1024))

            S_256, T_256 = [resize(np.array(im) / 255.0, (256, 256)) for im in [S, T]]
            T_mask = faceParsing_demo(self.faceParsing_model, T, convert_to_seg12=True,
                                      model_name=self.opts.faceParser_name) if target_mask is None else target_mask
            if verbose:
                Image.fromarray(T_mask).save(os.path.join(save_dir, "T_mask.png"))
                T_mask_vis = vis_parsing_maps(T, T_mask)
                Image.fromarray(T_mask_vis).save(os.path.join(save_dir, "T_mask_vis.png"))

            predictions = drive_source_demo(S_256, [T_256], self.generator, self.kp_detector, self.he_estimator,
                                            self.estimate_jacobian)
            predictions = [(pred * 255).astype(np.uint8) for pred in predictions]

            drivens = [GPEN_demo(pred[:, :, ::-1], self.GPEN_model, aligned=False) for pred in predictions]
            D = Image.fromarray(drivens[0][:, :, ::-1])

            D_mask = faceParsing_demo(self.faceParsing_model, D, convert_to_seg12=True,
                                      model_name=self.opts.faceParser_name)
            if verbose:
                Image.fromarray(D_mask).save(os.path.join(save_dir, "D_mask.png"))
                D_mask_vis = vis_parsing_maps(D, D_mask)
                Image.fromarray(D_mask_vis).save(os.path.join(save_dir, "D_mask_vis.png"))

            driven = transforms.Compose([TO_TENSOR, NORMALIZE])(D)
            driven = driven.to(self.opts.device).float().unsqueeze(0)
            driven_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(D_mask))
            driven_mask = (driven_mask * 255).long().to(self.opts.device).unsqueeze(0)
            driven_onehot = e4s_utils.labelMap2OneHot(driven_mask, num_cls=self.opts.num_seg_cls)

            target1 = transforms.Compose([TO_TENSOR, NORMALIZE])(T)
            target1 = target1.to(self.opts.device).float().unsqueeze(0)

            target_mask1 = transforms.Compose([TO_TENSOR])(Image.fromarray(T_mask))
            target_mask1 = (target_mask1 * 255).long().to(self.opts.device).unsqueeze(0)
            target_onehot = e4s_utils.labelMap2OneHot(target_mask1, num_cls=self.opts.num_seg_cls)

            driven_style_vector, _ = self.net.get_style_vectors(driven, driven_onehot)
            target_style_vector, _ = self.net.get_style_vectors(target1, target_onehot)
            if verbose:
                torch.save(driven_style_vector, os.path.join(save_dir, "D_style_vec.pt"))
                driven_style_codes = self.net.cal_style_codes(driven_style_vector)
                driven_face, _, structure_feats = self.net.gen_img(torch.zeros(1, 512, 32, 32).to(self.opts.device),
                                                                   driven_style_codes, driven_onehot)
                driven_face_image = e4s_utils.tensor2im(driven_face[0])
                driven_face_image.save(os.path.join(save_dir, "D_recon.png"))

                torch.save(target_style_vector, os.path.join(save_dir, "T_style_vec.pt"))
                target_style_codes = self.net.cal_style_codes(target_style_vector)
                target_face, _, structure_feats = self.net.gen_img(torch.zeros(1, 512, 32, 32).to(self.opts.device),
                                                                   target_style_codes, target_onehot)
                target_face_image = e4s_utils.tensor2im(target_face[0])
                target_face_image.save(os.path.join(save_dir, "T_recon.png"))

            swapped_msk, hole_map = swap_head_mask_revisit_considerGlass(D_mask, T_mask)

            if verbose:
                cv2.imwrite(os.path.join(save_dir, "swappedMask.png"), swapped_msk)
                swappped_one_hot = e4s_utils.labelMap2OneHot(
                    torch.from_numpy(swapped_msk).unsqueeze(0).unsqueeze(0).long(), num_cls=12)
                e4s_utils.tensor2map(swappped_one_hot[0]).save(os.path.join(save_dir, "swappedMaskVis.png"))

            comp_indices = set(range(self.opts.num_seg_cls)) - {0, 4, 11, 10}  # 10 glass, 8 neck
            swapped_style_vectors = swap_comp_style_vector(target_style_vector, driven_style_vector, list(comp_indices),
                                                           belowFace_interpolation=False)
            if verbose:
                torch.save(swapped_style_vectors, os.path.join(save_dir, "swapped_style_vec.pt"))

            swapped_msk = Image.fromarray(swapped_msk).convert('L')
            swapped_msk = transforms.Compose([TO_TENSOR])(swapped_msk)
            swapped_msk = (swapped_msk * 255).long().to(self.opts.device).unsqueeze(0)
            swapped_onehot = e4s_utils.labelMap2OneHot(swapped_msk, num_cls=self.opts.num_seg_cls)
            #
            swapped_style_codes = self.net.cal_style_codes(swapped_style_vectors)
            swapped_face, _, structure_feats = self.net.gen_img(torch.zeros(1, 512, 32, 32).to(self.opts.device),
                                                                swapped_style_codes, swapped_onehot)
            swapped_face_image = e4s_utils.tensor2im(swapped_face[0])

            outer_dilation = 5
            mask_bg = logical_or_reduce(*[swapped_msk == clz for clz in [0, 11, 4]])
            is_foreground = torch.logical_not(mask_bg)
            hole_index = hole_map[None][None] == 255
            is_foreground[hole_index[None]] = True
            foreground_mask = is_foreground.float()

            if self.opts.lap_bld:
                content_mask, border_mask, full_mask = create_masks(foreground_mask, outer_dilation=outer_dilation,
                                                                    operation='expansion')
            else:
                content_mask, border_mask, full_mask = create_masks(foreground_mask, outer_dilation=outer_dilation)

            content_mask = F.interpolate(content_mask, (1024, 1024), mode='bilinear', align_corners=False)
            content_mask_image = Image.fromarray(255 * content_mask[0, 0, :, :].cpu().numpy().astype(np.uint8))
            full_mask = F.interpolate(full_mask, (1024, 1024), mode='bilinear', align_corners=False)
            full_mask_image = Image.fromarray(255 * full_mask[0, 0, :, :].cpu().numpy().astype(np.uint8))

            if self.opts.lap_bld:
                content_mask = content_mask[0, 0, :, :, None].cpu().numpy()
                border_mask = F.interpolate(border_mask, (1024, 1024), mode='bilinear', align_corners=False)
                border_mask = border_mask[0, 0, :, :, None].cpu().numpy()
                border_mask = np.repeat(border_mask, 3, axis=-1)

                swapped_and_pasted = swapped_face_image * content_mask + T * (1 - content_mask)
                swapped_and_pasted = Image.fromarray(np.uint8(swapped_and_pasted))
                swapped_and_pasted = Image.fromarray(
                    blending(np.array(T), np.array(swapped_and_pasted), mask=border_mask))
            else:
                if outer_dilation == 0:
                    swapped_and_pasted = smooth_face_boundry(swapped_face_image, T, content_mask_image,
                                                             radius=outer_dilation)
                else:
                    swapped_and_pasted = smooth_face_boundry(swapped_face_image, T, full_mask_image,
                                                             radius=outer_dilation)

            pasted_image = swapped_and_pasted

            if pasted_image.mode == 'RGBA':
                pasted_image = pasted_image.convert('RGB')

                result.append(pasted_image)

        return result

    def forward(self, img_wtm_device):       
        source = img_wtm_device[1]
        target_wm = img_wtm_device[0]  
        img_size = source.shape[-1]
        resize_back = transforms.Resize((img_size, img_size))
        img_source = torch.roll(source, 1, 0)

        if len(self.opts.target_mask) != 0:
            target_mask = Image.open(self.opts.target_mask).convert("L")
            target_mask_seg12 = self.__celebAHQ_masks_to_faceParser_mask_detailed(target_mask)
        else:
            target_mask_seg12 = None

        swapped_face_wtm = self.faceSwapping_pipeline(
            denorm(img_source), 
            denorm(target_wm), 
            save_dir='temp_data/e4s_swap_images',
            target_mask = target_mask_seg12, 
            need_crop=False, 
            verbose=self.opts.verbose
        )
        resized_swapped_face_wtm = [resize_back(face) for face in swapped_face_wtm]
        swapped_face_wtm_tensors = torch.stack([transforms.ToTensor()(face) for face in resized_swapped_face_wtm], dim=0)
        swapped_face_wtm_tensors = norm(swapped_face_wtm_tensors.to(self.device))
        return swapped_face_wtm_tensors


# DiffSwap utilities
import argparse
import shutil
import dlib
import matplotlib.pyplot as plt
import matplotlib
import pickle
import json
import glob
import scipy
import random
from omegaconf import OmegaConf
from PIL import Image, ImageChops
from tqdm import tqdm
from model.DiffSwap.utils.portrait import Portrait
from einops import rearrange, repeat
from model.DiffSwap.utils.blending.blending_mask import gaussian_pyramid, laplacian_pyramid, laplacian_pyr_join, laplacian_collapse
from skimage import io
from imutils import face_utils
from torch.nn import DataParallel
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from model.DiffSwap.tests.faceswap_portrait import perform_swap
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class DiffSwapModel(nn.Module):
    def __init__(self, device, img_size):
        super(DiffSwapModel, self).__init__()
        self.root = 'DiffSwap'
        self.data_path = f'{self.root}/data/portrait'
        self.align_path = f'{self.data_path}/align'
        self.landmark_path = f'{self.data_path}/landmark'
        self.mask_path = f'{self.data_path}/mask'
        self.error_path= f'{self.data_path}/error_img.json'
        self.tgt_path = f'{self.align_path}/target'
        self.swap_path = f'{self.data_path}/swap_res'
        self.swap_res_repair_path = f'{self.data_path}/swap_res_repair'
        self.dst_dir = f'{self.data_path}/swap_res_ori'
        self.ori_tgt_path = f'{self.data_path}/target'
        self.affine_path=f'{self.data_path}/affines.json'
        self.img_size = img_size

        self.detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(
            f'{self.root}/checkpoints/shape_predictor_68_face_landmarks.dat'
        )
        self.config_path = f'{self.root}/configs/diffswap/default-project.yaml'
        self.checkpoint_path = f'{self.root}/checkpoints/diffswap.pth'
        self.affines_path = f'{self.data_path}/affines.json'
        
        self.device = device

        config = OmegaConf.load(self.config_path)

        print('Building DiffSwap model...')
        self.model = instantiate_from_config(config.model)
        self.model.init_from_ckpt(self.checkpoint_path)
       
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model = DataParallel(self.model)

        self.model.module.cond_stage_model.affine_crop = True
        self.model.module.cond_stage_model.swap = True

        self.ddim_sampler = DDIMSampler(self.model.module, tgt_scale=0.01)

        print('Model loaded successfully.')
    
    def jpg2png(self, pic_source, pic_target, a_name, b_name):
        for type in ['source', 'target']:
            if type == 'source':
                image = Image.open(pic_source)
                image = image.resize((256, 256), Image.Resampling.LANCZOS)

                img_png = a_name + '.png'
                os.makedirs(os.path.join(self.data_path, type), exist_ok=True)
                image.save(os.path.join(self.data_path, type, img_png))

                image = io.imread(os.path.join(self.data_path, type, img_png))
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
                cv2.imencode('.png', image)[1].tofile(os.path.join(self.data_path, type, img_png))
            else:
                image = Image.open(pic_target)
                image = image.resize((256, 256), Image.Resampling.LANCZOS)

                img_png = b_name + '.png'
                os.makedirs(os.path.join(self.data_path, type), exist_ok=True)
                image.save(os.path.join(self.data_path, type, img_png))

                image = io.imread(os.path.join(self.data_path, type, img_png))
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
                cv2.imencode('.png', image)[1].tofile(os.path.join(self.data_path, type, img_png))
                
    def get_detect(self, image, iter):
        for i in range(iter + 1):
            faces = self.detector(image, i)
            if len(faces) >= 1:
                break
        return faces   
    
    def get_lmk_ori(self):
        flag = 1
        all_lmk = {}
        for type in ['source', 'target']:
            all_lmk[type] = {}
            img_count = 0
            img_list = os.listdir(os.path.join(self.data_path, type))
            for img in tqdm(img_list, desc='image'):
                resize_flag = False
                image = cv2.imread(os.path.join(self.data_path, type, img))
                while image.shape[0] > 2000 or image.shape[1] > 2000:
                    resize_flag = True
                    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
                while image.shape[0] < 400 or image.shape[1] < 400:
                    resize_flag = True
                    image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.get_detect(imgray, 2)
                if len(faces) == 0:
                    print('error', type, img)
                    os.remove(os.path.join(self.data_path, type, img))
                    flag = 0
                    continue
                if len(faces) > 1:
                    print('> 1', type, img)
                    face = dlib.rectangle(faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom())
                    for i in range(1, len(faces)):
                        if abs(faces[i].right() - faces[i].left()) * abs(faces[i].top() - faces[i].bottom()) > abs(
                                face.right() - face.left()) * abs(face.top() - face.bottom()):
                            face = dlib.rectangle(faces[i].left(), faces[i].top(), faces[i].right(), faces[i].bottom())
                else:
                    face = faces[0]
                landmark = self.landmark_predictor(image, face)
                landmark = face_utils.shape_to_np(landmark)

                cv2.imwrite(os.path.join(self.data_path, type, img), image)
                all_lmk[type][img] = landmark

                img_count += 1

            print('type', type, 'img_count', img_count)
        if not os.path.exists(self.landmark_path):
            os.makedirs(self.landmark_path)
        pickle.dump(all_lmk, open(os.path.join(self.landmark_path, 'landmark_ori.pkl'), 'wb'))
        return flag
    
    def get_lmk_256(self):
        all_lmk = {}
        if os.path.exists(self.error_path):
            error_img = json.load(open(self.error_path, 'r'))
        else:
            error_img = {'source': [], 'target': []}
        for type in ['source', 'target']:
            all_lmk[type] = {}
            img_count = 0

            img_list = os.listdir(os.path.join(self.align_path, type))
            for img in tqdm(img_list, desc='image'):
                if os.path.exists(self.error_path):
                    if img in error_img[type]:
                        continue
                image = cv2.imread(os.path.join(self.align_path, type, img))
                imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                faces = self.get_detect(imgray, 2)
                if len(faces) == 0:
                    print('error', type, img)
                    error_img[type].append(img)
                    continue
                if len(faces) > 1:
                    print('> 1', type, img)
                    face = dlib.rectangle(faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom())
                    for i in range(1, len(faces)):
                        if abs(faces[i].right() - faces[i].left()) * abs(faces[i].top() - faces[i].bottom()) > abs(
                                face.right() - face.left()) * abs(face.top() - face.bottom()):
                            face = dlib.rectangle(faces[i].left(), faces[i].top(), faces[i].right(), faces[i].bottom())
                else:
                    face = faces[0]
                landmark = self.landmark_predictor(image, face)
                landmark = face_utils.shape_to_np(landmark)
                all_lmk[type][img] = landmark
                img_count += 1
            print('type', type, 'img_count', img_count)
        if not os.path.exists(self.landmark_path):
            os.makedirs(self.landmark_path)
        pickle.dump(all_lmk, open(os.path.join(self.landmark_path, 'landmark_256.pkl'), 'wb'))
        json.dump(error_img, open(self.error_path, 'w'), indent=4)
        
    def eval_lmk(self, mode):
        assert (mode in ['256', 'ori'])
        if mode == '256':
            data_path = os.path.join(self.data_path, 'align')
        landmarks = pickle.load(open(os.path.join(self.landmark_path, f'landmark_{mode}.pkl'), 'rb'))
        if os.path.exists(self.error_path):
            error_img = json.load(open(self.error_path, 'r'))
        for type in ['source', 'target']:
            eval_mode = 'eval_256' if mode == '256' else 'eval_ori'
            os.makedirs(os.path.join(self.landmark_path, eval_mode, type), exist_ok=True)
            img_list = os.listdir(os.path.join(self.data_path, type))
            for img in tqdm(img_list, desc='image'):
                if os.path.exists(self.error_path):
                    if img in error_img[type]:
                        continue
                image = cv2.imread(os.path.join(self.data_path, type, img))
                landmark = landmarks[type][img]
                for i in range(landmark.shape[0]):
                    cv2.circle(image, (round(landmark[i, 0]), round(landmark[i, 1])), 1, (0, 255, 0), -1, 8)

                cv2.imwrite(os.path.join(self.landmark_path, eval_mode, type, img), image)
    
    def crop_ffhq(self, output_size=256, transform_size=1024, enable_padding=False, rotate_level=True, random_shift=0, retry_crops=False):
        print('Recreating aligned images...')
        # Fix random seed for reproducibility
        np.random.seed(12345)
        landmarks = pickle.load(open('DiffSwap/data/portrait/landmark/landmark_ori.pkl', 'rb'))
        affine_all = {}
        for type in ['source', 'target']:
            img_count = 0
            affine_all[type] = {}

            img_list = os.listdir(os.path.join(self.data_path, type))
            for img in tqdm(img_list, desc='image'):
                lm = landmarks[type][img]
                lm_chin = lm[0: 17]  # left-right
                lm_eyebrow_left = lm[17: 22]  # left-right
                lm_eyebrow_right = lm[22: 27]  # left-right
                lm_nose = lm[27: 31]  # top-down
                lm_nostrils = lm[31: 36]  # top-down
                lm_eye_left = lm[36: 42]  # left-clockwise
                lm_eye_right = lm[42: 48]  # left-clockwise
                lm_mouth_outer = lm[48: 60]  # left-clockwise
                lm_mouth_inner = lm[60: 68]  # left-clockwise

                # Calculate auxiliary vectors.
                eye_left = np.mean(lm_eye_left, axis=0)
                eye_right = np.mean(lm_eye_right, axis=0)
                eye_avg = (eye_left + eye_right) * 0.5
                eye_to_eye = eye_right - eye_left
                mouth_left = lm_mouth_outer[0]
                mouth_right = lm_mouth_outer[6]
                mouth_avg = (mouth_left + mouth_right) * 0.5
                eye_to_mouth = mouth_avg - eye_avg

                # Choose oriented crop rectangle.
                if rotate_level:
                    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
                    x /= np.hypot(*x)
                    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
                    y = np.flipud(x) * [-1, 1]
                    c0 = eye_avg + eye_to_mouth * 0.1
                else:
                    x = np.array([1, 0], dtype=np.float64)
                    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
                    y = np.flipud(x) * [-1, 1]
                    c0 = eye_avg + eye_to_mouth * 0.1

                # Load in ori image.
                src_file = os.path.join(self.data_path, type, img)

                image = Image.open(src_file)
                quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])
                qsize = np.hypot(*x) * 2
                if random_shift != 0:
                    for _ in range(1000):
                        c = (c0 + np.hypot(*x) * 2 * random_shift * np.random.normal(0, 1, c0.shape))
                        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
                        crop = (
                        int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                        int(np.ceil(max(quad[:, 1]))))
                        if not retry_crops or not (
                                crop[0] < 0 or crop[1] < 0 or crop[2] >= image.width or crop[3] >= image.height):
                            break
                    else:
                        print('rejected image')
                        return
                # Shrink.
                shrink = int(np.floor(qsize / output_size * 0.5))
                if shrink > 1:
                    rsize = (int(np.rint(float(image.size[0]) / shrink)), int(np.rint(float(image.size[1]) / shrink)))
                    image = image.resize(rsize, Image.BICUBIC)
                    quad /= shrink
                    qsize /= shrink
                # Crop.
                border = max(int(np.rint(qsize * 0.1)), 3)
                crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                        int(np.ceil(max(quad[:, 1]))))
                crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, image.size[0]),
                        min(crop[3] + border, image.size[1]))
                IsCrop = False
                if crop[2] - crop[0] < image.size[0] or crop[3] - crop[1] < image.size[1]:
                    IsCrop = True
                    crop = tuple(map(round, crop))
                    image = image.crop(crop)  # (left, upper, right, lower)
                    quad -= crop[0:2]
                # Pad.
                pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                    int(np.ceil(max(quad[:, 1]))))
                pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - image.size[0] + border, 0),
                    max(pad[3] - image.size[1] + border, 0))
                if enable_padding and max(pad) > border - 4:
                    pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
                    image = np.pad(np.float32(image), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
                    h, w, _ = image.shape
                    y, x, _ = np.ogrid[:h, :w, :1]
                    mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                                    1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
                    blur = qsize * 0.02
                    image += (scipy.ndimage.gaussian_filter(image, [blur, blur, 0]) - image) * np.clip(mask * 3.0 + 1.0,
                                                                                                    0.0, 1.0)
                    image += (np.median(image, axis=(0, 1)) - image) * np.clip(mask, 0.0, 1.0)
                    image = Image.fromarray(np.uint8(np.clip(np.rint(image), 0, 255)), 'RGB')
                    quad += pad[:2]

                # Transform(with rotation)
                quad = (quad + 0.5).flatten()
                assert (abs((quad[2] - quad[0]) - (quad[4] - quad[6])) < 1e-6 and abs(
                    (quad[3] - quad[1]) - (quad[5] - quad[7])) < 1e-6)

                if IsCrop:
                    quad_new = [quad[0] + crop[0], quad[1] + crop[1], quad[2] + crop[0], quad[3] + crop[1],
                                quad[4] + crop[0], quad[5] + crop[1], quad[6] + crop[0], quad[7] + crop[1]]
                else:
                    quad_new = quad
                if shrink > 1:
                    quad_new *= shrink
                # print(f'quad_new: {quad_new}', 'type', type, 'img', img)
                affine_rev = ((256 * (quad_new[1] - quad_new[3])) / (
                            quad_new[0] * quad_new[3] - quad_new[1] * quad_new[2] - quad_new[0] * quad_new[5] + quad_new[
                        1] * quad_new[4] + quad_new[2] * quad_new[5] - quad_new[3] * quad_new[4]),
                            -(256 * (quad_new[0] - quad_new[2])) / (
                                        quad_new[0] * quad_new[3] - quad_new[1] * quad_new[2] - quad_new[0] * quad_new[
                                    5] + quad_new[1] * quad_new[4] + quad_new[2] * quad_new[5] - quad_new[3] * quad_new[
                                            4]),
                            (256 * (quad_new[0] * quad_new[3] - quad_new[1] * quad_new[2])) / (
                                        quad_new[0] * quad_new[3] - quad_new[1] * quad_new[2] - quad_new[0] * quad_new[
                                    5] + quad_new[1] * quad_new[4] + quad_new[2] * quad_new[5] - quad_new[3] * quad_new[
                                            4]),
                            -(256 * (quad_new[3] - quad_new[5])) / (
                                        quad_new[0] * quad_new[3] - quad_new[1] * quad_new[2] - quad_new[0] * quad_new[
                                    5] + quad_new[1] * quad_new[4] + quad_new[2] * quad_new[5] - quad_new[3] * quad_new[
                                            4]),
                            (256 * (quad_new[2] - quad_new[4])) / (
                                        quad_new[0] * quad_new[3] - quad_new[1] * quad_new[2] - quad_new[0] * quad_new[
                                    5] + quad_new[1] * quad_new[4] + quad_new[2] * quad_new[5] - quad_new[3] * quad_new[
                                            4]),
                            (256 * (quad_new[0] * quad_new[3] - quad_new[1] * quad_new[2] - quad_new[0] * quad_new[5] +
                                    quad_new[1] * quad_new[4])) / (
                                        quad_new[0] * quad_new[3] - quad_new[1] * quad_new[2] - quad_new[0] * quad_new[
                                    5] + quad_new[1] * quad_new[4] + quad_new[2] * quad_new[5] - quad_new[3] * quad_new[
                                            4]))
                affine_all[type][img] = affine_rev
                # use affine to transform image
                affine = (-(quad[0] - quad[6]) / transform_size, -(quad[0] - quad[2]) / transform_size, quad[0],
                        -(quad[1] - quad[7]) / transform_size, -(quad[1] - quad[3]) / transform_size, quad[1])
                image = image.transform((transform_size, transform_size), Image.AFFINE, affine,
                                        Image.BICUBIC)  # a, b, c, d, e, f

                if output_size < transform_size:
                    image = image.resize((output_size, output_size), Image.BICUBIC)

                # Save aligned image.
                dst_subdir = os.path.join(self.align_path, type)
                os.makedirs(dst_subdir, exist_ok=True)
                image.save(os.path.join(dst_subdir, img))

                img_count += 1
            print('type {} finished, processed {} images'.format(type, img_count))
        # All done.
        json.dump(affine_all, open(self.affine_path, 'w'), indent=4)
        
        
    def save_mask(self):
        device = 'cuda:0'
        batch_size = 16
        num_workers = 8
        dataset = Portrait(self.data_path)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

        print('start batch')
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            
            for mask_key in ['mask']:
                os.makedirs(os.path.join(self.mask_path, mask_key), exist_ok=True)
                mask = batch[mask_key]
                for i, image in enumerate(mask):
                    image = image.numpy() * 255
                    cv2.imwrite(os.path.join(self.mask_path, mask_key, batch['target'][i]), image)

    def repair_by_mask(self):
        gen_type_list = os.listdir(self.swap_path)
        for type in tqdm(gen_type_list):
            src_list = os.listdir(os.path.join(self.swap_path, type))
            mask_type = 'mask'
            print('type: {}, mask_type: {}'.format(type, mask_type))
            for src in tqdm(src_list, desc=type, leave=False):
                img_list = os.listdir(os.path.join(self.swap_path, type, src))
                for img in tqdm(img_list, desc=src, leave=False):
                    swap_img = cv2.imread(os.path.join(self.swap_path, type, src, img))
                    im1 = cv2.cvtColor(swap_img, cv2.COLOR_BGR2RGB)
                    tgt_img = cv2.imread(os.path.join(self.tgt_path, img))
                    im2 = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)
                    mask = matplotlib.image.imread(os.path.join(self.mask_path, mask_type, img))
                    im1, im2 = np.int32(im1), np.int32(im2)
                    mask = np.uint8(mask)

                    gp_1, gp_2 = [gaussian_pyramid(im) for im in [im1, im2]]
                    mask_gp = [cv2.resize(mask, (gp.shape[1], gp.shape[0])) for gp in gp_1]
                    lp_1, lp_2 = [laplacian_pyramid(gp) for gp in [gp_1, gp_2]]
                    lp_join = laplacian_pyr_join(lp_1, lp_2, mask_gp)
                    im_join = laplacian_collapse(lp_join)
                    np.clip(im_join, 0, 255, out=im_join)
                    im_join = np.uint8(im_join)

                    os.makedirs(os.path.join(self.swap_res_repair_path, type, src), exist_ok=True)
                    plt.imsave(os.path.join(self.swap_res_repair_path, type, src, img), im_join)
                    
    def paste(self, a_name, b_name):
        gen_type_list = os.listdir(self.swap_res_repair_path)
        affine_all = json.load(open(self.affines_path, 'r'))
        for type in tqdm(gen_type_list):
            src_list = os.listdir(os.path.join(self.swap_res_repair_path, type))
            for src in tqdm(src_list, desc=type, leave=False):
                img_list = os.listdir(os.path.join(self.swap_res_repair_path, type, src))
                for img in tqdm(img_list, desc=src, leave=False):
                    tgt_img = Image.open(os.path.join(self.ori_tgt_path, img)).convert('RGB')
                    gen_img = tgt_img.copy()
                    gen_img256 = Image.open(os.path.join(self.swap_res_repair_path, type, src, img)).convert('RGB')  # 256x256
                    mask = Image.new('RGBA', (256, 256), (255, 255, 255))
                    mask = mask.transform(tgt_img.size, Image.AFFINE, affine_all['target'][img], Image.BICUBIC)
                    affine_img = gen_img256.transform(tgt_img.size, Image.AFFINE, affine_all['target'][img], Image.BICUBIC)
                    gen_img.paste(affine_img, (0, 0), mask=mask)
                    os.makedirs(os.path.join(self.dst_dir, type, src), exist_ok=True)
                    gen_img = gen_img.resize((256, 256), Image.ANTIALIAS)
                    return gen_img

    def remove_files(self, directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)

    def forward(self, pic_source, pic_target, a_name, b_name):
        self.jpg2png(pic_source, pic_target, a_name, b_name)
        flag = self.get_lmk_ori()
        if flag == 0:
            self.remove_files(self.data_path)
            return None
        else:
            self.crop_ffhq()
            self.get_lmk_256()

            print('running face detection')
            os.system('bash DiffSwap/data_preprocessing/detection/run_detect_faces_portrait.sh')
            print('running mtcnn')
            os.system('python DiffSwap/data_preprocessing/detection/merge_mtcnn_portrait.py')
            print('obtain the parameters of affine transformation')
            os.system('python DiffSwap/data_preprocessing/align/face_align_portrait.py')

            self.save_mask()

            dataset = Portrait(self.data_path)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=32,
                num_workers=8,
                pin_memory=True,
                drop_last=False,
                        )
            print('start batch')
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                        print(f'Batch data {k} is on device: {batch[k].device}')

                perform_swap(self.model.module, batch, 'diffswap', self.ddim_sampler)
            self.repair_by_mask()
            swapped_img_wtm = self.paste(a_name, b_name)
            swapped_img_wtm = to_tensor(swapped_img_wtm).unsqueeze(0)
            resize_back = transforms.Resize((self.img_size, self.img_size))
            swapped_img_wtm = resize_back(swapped_img_wtm)
            print(swapped_img_wtm.shape)
            self.remove_files(self.data_path)
        return swapped_img_wtm


## StarGAN utilities
from munch import Munch
from torch.backends import cudnn
from model.stargan.para import SwapFacePipelineOptions as StarGANSwapOptions
from model.stargan.core.data_loader import get_train_loader
from model.stargan.core.data_loader import get_test_loader
from model.stargan.core.solver import Solver
import torchvision.utils as vutils


class StarGANModel(nn.Module):

    def __init__(self, device):
        super(StarGANModel, self).__init__()
        self.args = StarGANSwapOptions().parse()
        self.device = device
        self.solver = Solver(self.args)

    def denormalize(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def generate_images(self, src_image1, target_image1, target_label1):
        x_fake = self.solver.sample(src_image1, target_image1, target_label1)
        return x_fake

    def forward(self, img_wtm_device):
        img_wtm = img_wtm_device[0]
        img = img_wtm_device[1]

        img_size = img_wtm.shape[-1]
        resize = transforms.Resize((256, 256))
        resize_back = transforms.Resize((img_size, img_size))
        img, img_wtm = resize(img), resize(img_wtm)
        label = torch.tensor([0]).to(self.device)
        
        img_source = torch.roll(img, 1, 0)
        deepfake_face_wtm = self.generate_images(img_source, img_wtm, label)
        return resize_back(deepfake_face_wtm)


# StyleMask utilities
import face_alignment
from argparse import Namespace
from StyleMask.libs.models.StyleGAN2.model import Generator as StyleGAN2Generator
from StyleMask.libs.models.mask_predictor import MaskPredictor
from StyleMask.libs.utilities.utils import make_noise, generate_image, generate_new_stylespace, save_image, save_grid, get_files_frompath
from StyleMask.libs.utilities.stylespace_utils import decoder
from StyleMask.libs.configs.config_models import stylegan2_ffhq_1024
from StyleMask.libs.utilities.utils_inference import preprocess_image, invert_image
from StyleMask.libs.utilities.image_utils import image_to_tensor
from StyleMask.libs.models.inversion.psp import pSp


class StyleMaskModel(nn.Module):

    def __init__(self, device):
        super(StyleMaskModel, self).__init__()
        self.device = device
        self.output_path = './results'
        arguments_json = os.path.join(self.output_path, 'arguments.json')
        self.masknet_path = 'StyleMask/pretrained_models/mask_network_1024.pt'
        self.image_resolution = 1024
        self.dataset = 'celeba-hq'

        self.num_pairs = 4
        self.resize_image = True
        self.load_models(True)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

    def load_models(self, inversion):
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.generator_path = stylegan2_ffhq_1024['gan_weights']
        self.channel_multiplier = stylegan2_ffhq_1024['channel_multiplier']
        self.split_sections = stylegan2_ffhq_1024['split_sections']
        self.stylespace_dim = stylegan2_ffhq_1024['stylespace_dim']

        if os.path.exists(self.generator_path):
            print('----- Load generator from {} -----'.format(self.generator_path))
            self.G = StyleGAN2Generator(self.image_resolution, 512, 8, channel_multiplier=self.channel_multiplier)
            self.G.load_state_dict(torch.load(self.generator_path)['g_ema'], strict=True)
            self.G.cuda().eval()
            # use truncation
            self.truncation = 0.7
            self.trunc = self.G.mean_latent(4096).detach().clone()
        else:
            print(
                'Please download the pretrained model for StyleGAN2 generator and save it into ./pretrained_models path')
            exit()

        if os.path.exists(self.masknet_path):
            print('----- Load mask network from {} -----'.format(self.masknet_path))
            ckpt = torch.load(self.masknet_path, map_location=torch.device('cpu'))
            self.num_layers_control = ckpt['num_layers_control']
            self.mask_net = nn.ModuleDict({})
            for layer_idx in range(self.num_layers_control):
                network_name_str = 'network_{:02d}'.format(layer_idx)

                # Net info
                stylespace_dim_layer = self.split_sections[layer_idx]
                input_dim = stylespace_dim_layer
                output_dim = stylespace_dim_layer
                inner_dim = stylespace_dim_layer

                network_module = MaskPredictor(input_dim, output_dim, inner_dim=inner_dim)
                self.mask_net.update({network_name_str: network_module})
            self.mask_net.load_state_dict(ckpt['mask_net'])
            self.mask_net.cuda().eval()
        else:
            print('Please download the pretrained model for Mask network and save it into ./pretrained_models path')
            exit()

        if inversion:
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda')
            # Load inversion model only when the input is image.
            self.encoder_path = stylegan2_ffhq_1024['e4e_inversion_model']
            print('----- Load e4e encoder from {} -----'.format(self.encoder_path))
            ckpt = torch.load(self.encoder_path, map_location='cpu')
            opts = ckpt['opts']
            opts['output_size'] = self.image_resolution
            opts['checkpoint_path'] = self.encoder_path
            opts['device'] = self.device
            opts['channel_multiplier'] = self.channel_multiplier
            opts['dataset'] = self.dataset
            opts = Namespace(**opts)
            self.encoder = pSp(opts)
            self.encoder.cuda().eval()

    def reenact_pair(self, source_code, target_code):
        with torch.no_grad():
            # Get source style space
            source_img, style_source, w_source, noise_source = generate_image(
                self.G, source_code, self.truncation,
                self.trunc, self.image_resolution,
                self.split_sections,
                input_is_latent=self.input_is_latent,
                return_latents=True,
                resize_image=self.resize_image
            )
            # Get target style space
            target_img, style_target, w_target, noise_target = generate_image(
                self.G, target_code, self.truncation,
                self.trunc, self.image_resolution,
                self.split_sections,
                input_is_latent=self.input_is_latent,
                return_latents=True,
                resize_image=self.resize_image
            )
            # Get reenacted image
            masks_per_layer = []
            for layer_idx in range(self.num_layers_control):
                network_name_str = 'network_{:02d}'.format(layer_idx)
                style_source_idx = style_source[layer_idx]
                style_target_idx = style_target[layer_idx]
                styles = style_source_idx - style_target_idx
                mask_idx = self.mask_net[network_name_str](styles)
                masks_per_layer.append(mask_idx)

            mask = torch.cat(masks_per_layer, dim=1)
            style_source = torch.cat(style_source, dim=1)
            style_target = torch.cat(style_target, dim=1)

            new_style_space = generate_new_stylespace(style_source, style_target, mask,
                                                      num_layers_control=self.num_layers_control)
            new_style_space = list(
                torch.split(tensor=new_style_space, split_size_or_sections=self.split_sections, dim=1))
            reenacted_img = decoder(self.G, new_style_space, w_source, noise_source, resize_image=self.resize_image)

        return source_img, target_img, reenacted_img

    def run(self, source, target):
        inv_image, source_code = invert_image(source, self.encoder, self.G, self.truncation, self.trunc)
        inv_image, target_code = invert_image(target, self.encoder, self.G, self.truncation, self.trunc)
        self.input_is_latent = True
        source_img, target_img, reenacted_img = self.reenact_pair(source_code, target_code)
   
        return source_img, target_img, reenacted_img

    def forward(self, img_wtm_device):
        img_wtm = img_wtm_device[0]
        img = img_wtm_device[1]
        img_size = img_wtm.shape[-1]
        resize = transforms.Resize((256, 256))
        resize_back = transforms.Resize((img_size, img_size))
        img, img_wtm = resize(img), resize(img_wtm)
        img_source = torch.roll(img, 1, 0)
        _, _, swapped_face_wtm = self.run(img_source, img_wtm)
        swapped_face_wtm = resize_back(swapped_face_wtm)
        return swapped_face_wtm


# HyperReenact utilities
from model.HyperReenact.libs.face_models.landmarks_estimation import LandmarksEstimation
from model.HyperReenact.libs.models.pose_encoder import DECAEncoder
from model.HyperReenact.libs.models.appearance_encoder import ArcFaceEncoder
from model.HyperReenact.libs.models.encoders.psp_encoders import Encoder4Editing
from model.HyperReenact.libs.models.hypernetwork_reenact import Hypernetwork_reenact
from model.HyperReenact.libs.DECA.decalib.datasets import datasets
from model.HyperReenact.libs.utilities.image_utils import *
from model.HyperReenact.libs.configs.config_models import *
from model.HyperReenact.libs.utilities.utils import *
import argparse
from argparse import Namespace
import random
root_path = os.getcwd()
random.seed(0)


class HyperReenactModel(nn.Module):

    def __init__(self, device):
        super(HyperReenactModel, self).__init__()
        # self.args = args
        self.device = device
        self.output_path = './results'
        make_path(self.output_path)

        self.model_path = 'HyperReenact/pretrained_models/hypernetwork.pt'

        self.save_grids = True
        self.save_images = True
        self.save_video = False

        self.image_resolution = model_arguments['image_resolution']
        self.deca_layer = model_arguments['deca_layer']
        self.arcface_layer = model_arguments['arcface_layer']
        self.pose_encoder_path = model_arguments['pose_encoder_path']
        self.app_encoder_path = model_arguments['app_encoder_path']
        self.e4e_path = model_arguments['e4e_path']
        self.sfd_detector_path = model_arguments['sfd_detector_path']

        self.load_models()

    def load_auxiliary_models(self):
        self.landmarks_est = LandmarksEstimation(type='2D', path_to_detector=self.sfd_detector_path)

        ################ Pose encoder ################
        print('********* Upload pose encoder *********')
        self.pose_encoder = DECAEncoder(layer=self.deca_layer).to(self.device)  # resnet50 pretrained for DECA eval mode
        self.posedata = datasets.TestData()
        ckpt = torch.load(self.pose_encoder_path, map_location='cpu')
        d = ckpt['E_flame']
        self.pose_encoder.load_state_dict(d)
        self.pose_encoder.eval()
        ##############################################

        ############# Appearance encoder #############
        print('********* Upload appearance encoder *********')
        self.appearance_encoder = ArcFaceEncoder(num_layer=self.arcface_layer).to(self.device)  # ArcFace model
        ckpt = torch.load(self.app_encoder_path, map_location='cpu')
        d_filt = {'facenet.{}'.format(k): v for k, v in ckpt.items()}
        self.appearance_encoder.load_state_dict(d_filt)
        self.appearance_encoder.eval()
        #############################################

        print('********* Upload Encoder4Editing *********')
        self.encoder = Encoder4Editing(50, 'ir_se', self.image_resolution).to(self.device)
        ckpt = torch.load(self.e4e_path)
        self.encoder.load_state_dict(ckpt['e'])
        self.encoder.eval()

    def load_models(self):

        self.load_auxiliary_models()

        print('********* Upload HyperReenact *********')
        opts = {}
        opts['root_path'] = root_path
        opts['device'] = self.device
        opts['deca_layer'] = self.deca_layer
        opts['arcface_layer'] = self.arcface_layer
        opts['checkpoint_path'] = self.model_path
        opts['output_size'] = self.image_resolution
        opts['channel_multiplier'] = model_arguments['channel_multiplier']
        opts['layers_to_tune'] = model_arguments['layers_to_tune']
        opts['mode'] = model_arguments['mode']
        opts['stylegan_weights'] = model_arguments['generator_weights']

        opts = Namespace(**opts)
        self.net = Hypernetwork_reenact(opts).to(self.device)
        self.net.eval()

        if self.net.latent_avg is None:
            self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[0].detach()

        self.truncation = 0.7
        self.trunc = self.net.decoder.mean_latent(4096).detach().clone()

    def get_identity_embeddings(self, image):

        landmarks = get_landmarks(image, self.landmarks_est)
        id_hat, f_app = self.appearance_encoder(image, landmarks)  # f_app 256 x 14 x 14 and id_hat 512

        return id_hat, f_app

    def get_pose_embeddings(self, image):
        image_pose = image.clone()
        image_prepro = torch.zeros(image_pose.shape[0], 3, 224, 224).cuda()
        for k in range(image_pose.shape[0]):
            min_val = -1
            max_val = 1
            image_pose[k].clamp_(min=min_val, max=max_val)
            image_pose[k].add_(-min_val).div_(max_val - min_val + 1e-5)
            image_pose[k] = image_pose[k].mul(255.0).add(0.0)
            image_prepro_, error_flag = self.posedata.get_image_tensor(image_pose[k])
            image_prepro[k] = image_prepro_
        pose_hat, f_pose = self.pose_encoder(image_prepro)  # 512, 28, 28

        return pose_hat, f_pose

    def run_reenactment(self, source, target):
        source_code = self.encoder(source)
        id_hat, f_app = self.get_identity_embeddings(target)
        pose_hat, f_pose = self.get_pose_embeddings(source)
        reenacted_image, _, _ = self.net.forward(
            f_pose=f_pose,
            f_app=f_app,
            codes=source_code,
            truncation=self.truncation,
            trunc=self.trunc,
            return_latents=True,
            return_weight_deltas_and_codes=True
        )
        return reenacted_image

    def forward(self, img_wtm_device):
        img_wtm = img_wtm_device[0]
        img = img_wtm_device[1]
        img_size = img_wtm.shape[-1]
        
        resize = transforms.Resize((256, 256))
        resize_back = transforms.Resize((img_size, img_size))

        img, img_wtm = resize(img), resize(img_wtm)
        img_source = torch.roll(img, 1, 0)
        swapped_face_wtm = self.run_reenactment(img_source, img_wtm)
        swapped_face_wtm = resize_back(swapped_face_wtm)
        return swapped_face_wtm
