import torch
import torch.nn as nn
import numpy as np
import cv2
import lpips
import kornia

from model.network import FractalForensics, WatermarkDecoder
from model.common_manipulations import Manipulation
from project_utils import denorm, wtm_error_rate, patch_accuracy, patch_accuracy_per_image


class Tester(object):

    def __init__(self, configs, device):
        super().__init__()
        self.img_size = configs.img_size
        self.wtm_size = configs.wtm_size
        self.latent_channels = configs.latent_channels
        self.img_blocks = configs.img_blocks
        self.wtm_blocks = configs.wtm_size
        self.rec_blocks = configs.rec_blocks
        self.dec_blocks = configs.dec_blocks

        self.batch_size = configs.batch_size
        self.device = device

        self.ff_model = FractalForensics(
            self.img_size,
            self.wtm_size,
            self.latent_channels,
            self.img_blocks,
            self.wtm_blocks,
            self.rec_blocks
        ).to(self.device)
        self.decoder = WatermarkDecoder(
            latent_channels=self.latent_channels,
            num_blocks=self.dec_blocks
        ).to(self.device)
        self.manipulation = Manipulation(configs.manipulation_layers)

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1 and self.device != 'cpu':
            print('Using', torch.cuda.device_count(), 'GPUs ...')
            self.ff_model = nn.DataParallel(self.ff_model)
            self.decoder = nn.DataParallel(self.decoder)

        self.metric_LPIPS = lpips.LPIPS(net='alex', pretrained=True).to(self.device)
        self.metric_LPIPS.eval()
        self.ff_model.eval()
        self.decoder.eval()

    def test_one_manipulation(self, imgs, wtms, check_visual=False, per_item=False):
        with torch.no_grad():
            imgs, wtms = imgs.to(self.device), wtms.to(self.device)
            imgs_wtm = self.ff_model(imgs, wtms)
            imgs_wtm_manipulated = self.manipulation([imgs_wtm, imgs, self.device])
            wtms_rec = self.decoder(imgs_wtm_manipulated)

            error_rate = wtm_error_rate(wtms, wtms_rec)
            accuracy = 1.0 - error_rate
            accuracy_patch = patch_accuracy(wtms, wtms_rec)

            if check_visual:
                psnr, ssim, lpips = Tester.visual_metrics(imgs, imgs_wtm, self.metric_LPIPS)
                if per_item:
                    return accuracy, accuracy_patch, psnr, ssim, lpips, patch_accuracy_per_image(wtms, wtms_rec)
                return accuracy, accuracy_patch, psnr, ssim, lpips
            else:
                if per_item:
                    return accuracy, accuracy_patch, patch_accuracy_per_image(wtms, wtms_rec)
                return accuracy, accuracy_patch

    def reset_manipulation(self, manipulation_layers):
        self.manipulation = Manipulation([manipulation_layers]).to(self.device)
        if self.num_gpus > 1:
            self.manipulation = nn.DataParallel(self.manipulation)
        self.manipulation.eval()
        print('Manipulation layer set to {}.'.format(manipulation_layers))

    def test_batch_deepfake(self, imgs, wtms, deepfake_model, per_item=False):
        with torch.no_grad():
            imgs, wtms = imgs.to(self.device), wtms.to(self.device)
            imgs_wtm = self.ff_model(imgs, wtms)
            swapped_img_wtm = deepfake_model([imgs_wtm, imgs, self.device])
            wtms_rec = self.decoder(swapped_img_wtm)

            error_rate = wtm_error_rate(wtms, wtms_rec)
            accuracy = 1.0 - error_rate
            accuracy_patch = patch_accuracy(wtms, wtms_rec)
            if per_item:
                return accuracy, accuracy_patch, patch_accuracy_per_image(wtms, wtms_rec)
        return accuracy, accuracy_patch
    
    # InfoSwap needs a sequential pipeline to perform normally (avoid purple faces). 
    def test_batch_infoswap(self, imgs, wtms, fnames, infoswap_model, per_item=False):
        with torch.no_grad():
            imgs, wtms = imgs.to(self.device), wtms.to(self.device)
            imgs_wtm = self.ff_model(imgs, wtms)
            imgs_denorm, imgs_wtm_denorm = denorm(imgs), denorm(imgs_wtm)
            
            # Save images.
            saved_img_names = []
            swapped_img_wm_list = []
            for i in range(imgs.shape[0]):
                img_name = fnames[i].split('.')[0]
                saved_img_names.append(img_name)
                source = np.array(imgs_denorm[i].detach().cpu().permute(1, 2, 0))[..., ::-1] * 255
                source = source.clip(0, 255).astype(np.uint8)
                wm = np.array(imgs_wtm_denorm[i].detach().cpu().permute(1, 2, 0))[..., ::-1] * 255
                wm = wm.clip(0, 255).astype(np.uint8)
                cv2.imwrite(f"/temp_data/infoswap/{img_name}_source.jpg", source)
                cv2.imwrite(f"/temp_data/infoswap/{img_name}_wm.jpg", wm)
            
            # Load images for face swapping. 
            N = len(saved_img_names)
            for j in range(N):
                source_name = saved_img_names[(j + 1) % N]
                target_wm_name = saved_img_names[j]
                source_path = f"/temp_data/infoswap/{source_name}_source.jpg"
                target_wm_path = f"/temp_data/infoswap/{target_wm_name}_wm.jpg"
                swapped_img_wm_i = infoswap_model(source_path, target_wm_path)
                swapped_img_wm_list.append(swapped_img_wm_i)
            
            swapped_img_wm_batch = torch.cat(swapped_img_wm_list, dim=0)
            wtms_rec = self.decoder(swapped_img_wm_batch)

            error_rate = wtm_error_rate(wtms, wtms_rec)
            accuracy = 1.0 - error_rate
            accuracy_patch = patch_accuracy(wtms, wtms_rec)
            if per_item:
                return accuracy, accuracy_patch, patch_accuracy_per_image(wtms, wtms_rec)
        return accuracy, accuracy_patch
    
    # DiffSwap needs a sequential pipeline to perform normally (avoid empty return value). 
    def test_batch_diffswap(self, imgs, wtms, fnames, diffswap_model, per_item=False):
        with torch.no_grad():
            imgs, wtms = imgs.to(self.device), wtms.to(self.device)
            imgs_wm = self.ff_model(imgs, wtms)
            imgs_denorm, imgs_wm_denorm = denorm(imgs), denorm(imgs_wm)

            # Save images.
            saved_img_names = []
            swapped_img_wm_list = []
            valid_indices = []
            for i in range(imgs.shape[0]):
                img_name = fnames[i].split('.')[0]
                saved_img_names.append(img_name)
                img = np.array(imgs_denorm[i].detach().cpu().permute(1, 2, 0))[..., ::-1] * 255
                img = img.clip(0, 255).astype(np.uint8)
                wm_img = np.array(imgs_wm_denorm[i].detach().cpu().permute(1, 2, 0))[..., ::-1] * 255
                wm_img = wm_img.clip(0, 255).astype(np.uint8)
                cv2.imwrite(f"/temp_data/diffswap_img/{img_name}_source.jpg", img)
                cv2.imwrite(f"/temp_data/diffswap_img/{img_name}_wm.jpg", wm_img)

            # Load images for face swapping. 
            N = len(saved_img_names)
            for j in range(N):
                source_name = saved_img_names[(j + 1) % N]
                target_wm_name = saved_img_names[j]
                source_path = f"/temp_data/diffswap_img/{source_name}_source.jpg"
                target_wm_path = f"/temp_data/diffswap_img/{target_wm_name}_wm.jpg"

                swapped_face_wm = diffswap_model(source_path, target_wm_path, source_name, target_wm_name)
                if swapped_face_wm is None:
                    print(f"[Warning] swap failed for {source_name} -> skip")
                    continue
                swapped_img_wm_list.append(swapped_face_wm)
                valid_indices.append(j)

            wtms_valid = wtms[valid_indices]
            swapped_img_wm_batch = torch.cat(swapped_img_wm_list, dim=0)
            
            wtms_rec = self.decoder(swapped_img_wm_batch.to(self.device))

            error_rate = wtm_error_rate(wtms_valid, wtms_rec)
            accuracy = 1.0 - error_rate
            accuracy_patch = patch_accuracy(wtms_valid, wtms_rec)
            if per_item:
                return accuracy, accuracy_patch, patch_accuracy_per_image(wtms_valid , wtms_rec)
        return accuracy, accuracy_patch

    def load_model(self, encoder_path, decoder_path):
        if self.num_gpus > 1:
            self.ff_model.module.load_state_dict(torch.load(encoder_path))
            self.decoder.module.load_state_dict(torch.load(decoder_path))
        else:
            self.ff_model.load_state_dict(torch.load(encoder_path))
            self.decoder.load_state_dict(torch.load(decoder_path))
        print('Loaded model weights from {} and {}.'.format(encoder_path, decoder_path))

    @staticmethod
    def visual_metrics(imgs_raw, imgs_wtm, lpips_metric):
        psnr = kornia.losses.psnr_loss(denorm(imgs_raw), denorm(imgs_wtm), 1.) * (-1)
        ssim = 1 - kornia.losses.ssim_loss(denorm(imgs_raw), denorm(imgs_wtm), window_size=5, reduction='mean')
        lpips = lpips_metric(imgs_wtm, imgs_raw).mean()
        return psnr.item(), ssim.item(), lpips.item()

