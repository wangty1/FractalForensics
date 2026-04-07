import warnings

warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated since 0.13")
warnings.filterwarnings("ignore",
                        message="Arguments other than a weight enum or `None` for 'weights' are deprecated")
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import time
import torch
from tqdm import tqdm

from project_utils import JsonConfig, format_time, make_loader
from tester import Tester


def test_common_all(configs, device, per_item=False):
    tester, test_img_loader, test_wtm_loader, log_path = prepare_model(configs, device)
    manipulation_lst = ['Resize(0.5)', 'GaussianNoise()', 'GaussianBlur(2,3)', 'MedBlur(3)', 'Jpeg(50)', 'Identity()']
    accuracy_dict = {}
    accuracy_patch_dict = {}
    if per_item:
        rate_dict = {}
    for m in manipulation_lst:
        accuracy_dict[m] = 0.0
        accuracy_patch_dict[m] = 0.0
        if per_item:
            rate_dict[m] = []
    psnr = 0.0
    ssim = 0.0
    lpips = 0.0

    start_time = time.time()
    for m in manipulation_lst:
        print("Testing with {}.".format(m))
        check_visual = True if m == 'Identity()' else False
        batch_count = 0
        tester.reset_manipulation(m)
        wtm_iter = iter(test_wtm_loader)
        for imgs in tqdm(test_img_loader):
            wtms = next(wtm_iter)
            ret_lst = tester.test_one_manipulation(imgs, wtms, check_visual=check_visual, per_item=per_item)
            accuracy_dict[m] += ret_lst[0]
            accuracy_patch_dict[m] += ret_lst[1]
            if per_item:
                rate_dict[m].extend(ret_lst[-1])
            batch_count += 1
            if check_visual:
                psnr += ret_lst[2]
                ssim += ret_lst[3]
                lpips += ret_lst[4]
        psnr /= batch_count
        ssim /= batch_count
        lpips /= batch_count
        accuracy_dict[m] /= batch_count
        accuracy_patch_dict[m] /= batch_count
    log_text = "Finished testing in {}.\n".format(format_time(time.time() - start_time))
    log_text += "All bit accuracy. \n"
    for m, acc in accuracy_dict.items():
        log_text += "{}: {}\n".format(m, acc)
    log_text += "Patch accuracy. \n"
    for m, acc in accuracy_patch_dict.items():
        log_text += "{}: {}\n".format(m, acc)
    log_text += "Visual quality. \n"
    log_text += "PSNR: {}\n".format(psnr)
    log_text += "SSIM: {}\n".format(ssim)
    log_text += "LPIPS: {}\n".format(lpips)
    print(log_text)
    with open(os.path.join(log_path, 'test_common_log.txt'), 'a') as f:
        f.write(log_text)

    if per_item:  # For AUC computation.
        for m, lst in rate_dict.items():
            with open(os.path.join(configs.log_path, m + "_scores.txt"), "w") as f:
                for score in lst:
                    f.write(f"{score:.6f}\n")


def initialize_deepfake_model(configs, device, model_name, mode='batch'):
    if model_name == 'SimSwap':
        from model.deepfakes import SimSwapModel
        deepfake_model = SimSwapModel(configs.img_size).to(device)
    elif model_name == 'InfoSwap':
        from model.deepfakes import InfoSwapModel
        deepfake_model = InfoSwapModel(device, configs.img_size).to(device)
    elif model_name == 'UniFace':
        from model.deepfakes import UniFaceModel
        deepfake_model = UniFaceModel(device).to(device)
    elif model_name == 'E4S':
        from model.deepfakes import E4SModel
        deepfake_model = E4SModel(device).to(device)
    elif model_name == 'DiffSwap':
        from model.deepfakes import DiffSwapModel
        deepfake_model = DiffSwapModel(device, configs.img_size).to(device)
    elif model_name == 'StarGAN':
        from model.deepfakes import StarGANModel
        deepfake_model = StarGANModel(device).to(device)
    elif model_name == 'StyleMask':
        from model.deepfakes import StyleMaskModel
        deepfake_model = StyleMaskModel(device).to(device)
    elif model_name == 'HyperReenact':
        from model.deepfakes import HyperReenactModel
        deepfake_model = HyperReenactModel(device).to(device)
    
    deepfake_model.eval()
    if mode == 'batch':
        num_gpus = torch.cuda.device_count()
        # E4S and DiffSwap original code do not fit parallel well. 
        if num_gpus > 1 and device != 'cpu' and model_name != 'E4S' and model_name != 'DiffSwap':
            deepfake_model = torch.nn.DataParallel(deepfake_model)
    return deepfake_model


def test_deepfake(configs, device, model_name, per_item=False):
    tester, test_img_loader, test_wtm_loader, log_path = prepare_model(configs, device)
    wtm_iter = iter(test_wtm_loader)
    batch_count = 0
    accuracy = 0.0
    accuracy_patch = 0.0

    if per_item:
        rate_lst = []

    deepfake_model = initialize_deepfake_model(configs, device, model_name)
    print("Testing with {}. \n".format(model_name))
    for imgs, fnames in tqdm(test_img_loader):
        wtms = next(wtm_iter)

        if model_name == 'DiffSwap':  # Special arrangement due to original model code. 
            ret_lst = tester.test_batch_diffswap(imgs, wtms, fnames, deepfake_model, per_item)
        elif model_name == 'InfoSwap':  # Special arrangement due to original model code. 
            ret_lst = tester.test_batch_infoswap(imgs, wtms, fnames, deepfake_model, per_item)
        else:
            ret_lst = tester.test_batch_deepfake(imgs, wtms, deepfake_model, per_item)

        accuracy += ret_lst[0]
        accuracy_patch += ret_lst[1]
        if per_item:
            rate_lst.extend(ret_lst[2].tolist())
        batch_count += 1

    print('Accuracy: {:.4f}'.format(accuracy / batch_count))
    print('Accuracy Patch: {:.4f}'.format(accuracy_patch / batch_count))
    if per_item:  # For AUC computation.
        with open(os.path.join(log_path, model_name + "_scores.txt"), "w") as f:
            for score in rate_lst:
                f.write(f"{score:.6f}\n")


def prepare_model(configs, device, loader=True):
    tester = Tester(configs, device)
    encoder_path = os.path.join(configs.weight_path, 'encoder.pth')
    decoder_path = os.path.join(configs.weight_path, 'decoder.pth')
    tester.load_model(encoder_path, decoder_path)
    log_path = configs.log_path
    if loader:
        test_img_loader, test_wtm_loader = make_loader(configs, 'test', False)
        return tester, test_img_loader, test_wtm_loader, log_path
    else:
        return tester, log_path


if __name__ == '__main__':
    configs = JsonConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test against common image operations. 
    configs.load_json_file('./configurations/test_common.json')
    configs.write_to_log(configs.log_path, 'test_common_log.txt')
    test_common_all(configs, device)

    # Test against Deepfake manipulations. 
    configs.load_json_file('./configurations/test_deepfake.json')
    configs.write_to_log(configs.log_path, 'test_deepfake_log.txt')
    test_deepfake(configs, device, 'SimSwap')
    # Change the model name to test against different Deepfake manipulations. 
    test_deepfake(configs, device, 'StarGAN')
