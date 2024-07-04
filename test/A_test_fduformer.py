import os.path
import logging
import argparse

import numpy as np
from datetime import datetime
from collections import OrderedDict
# from scipy.io import loadmat

import torch

from utils import utils_logger
from utils import utils_image as util
import utils

'''
Spyder (Python 3.6)
PyTorch 1.1.0
Windows 10 or Linux

Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/KAIR
        https://github.com/cszn/DnCNN

@article{zhang2017beyond,
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017},
  publisher={IEEE}
}

% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; github: https://github.com/cszn)

by Kai Zhang (12/Dec./2019)
'''

"""
# --------------------------------------------
|--model_zoo          # model_zoo
   |--dncnn_15        # model_name
   |--dncnn_25
   |--dncnn_50
   |--dncnn_gray_blind
   |--dncnn_color_blind
   |--dncnn3
|--testset            # testsets
   |--set12           # testset_name
   |--bsd68
   |--cbsd68
|--results            # results
   |--set12_dncnn_15  # result_name = testset_name + '_' + model_name
   |--set12_dncnn_25
   |--bsd68_dncnn_15
# --------------------------------------------
"""


def main():
    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    parser = argparse.ArgumentParser(description='Image motion deblurring evaluation on RealBlur_J/RealBlur_R')
    parser.add_argument('--testset_name', type=str, default='set12', help='test set, bsd68 | set12')
    parser.add_argument('--testsets', type=str, default='testsets', help='path of testing folder')

    parser.add_argument('--input_dir', default='/data1/wangzd/datasets/deblurring', type=str,
                        help='Directory of validation images')
    parser.add_argument('--result_dir', default='/data1/wangzd/uformer_cvpr/results_release/deblurring/', type=str,
                        help='Directory for results')
    parser.add_argument('--dataset', default='RealBlur_J,RealBlur_R', type=str, help='Test Dataset')
    parser.add_argument('--weights',
                        default='../train/logs/FDUformer0001/motiondeblur/SIDD/Uformer_/models/model_epoch_160.pth',
                        type=str, help='Path to weights')
    parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--arch', default='FDUformer', type=str, help='arch')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
    parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
    parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')
    parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
    parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
    parser.add_argument('--query_embed', action='store_true', default=False, help='query embedding for the decoder')
    parser.add_argument('--dd_in', type=int, default=1, help='dd_in')

    # args for vit
    parser.add_argument('--vit_dim', type=int, default=512, help='vit hidden_dim')
    parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
    parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
    parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
    parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
    parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
    parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
    parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')
    parser.add_argument('--model_pool', type=str, default='../train/logs/UNet/motiondeblur/SIDD/UNet_/models', help='share vit module')
    parser.add_argument('--train_ps', type=int, default=512, help='patch size of training sample')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    result_name = 'uformer' + '_' + 'bestmodel'     # fixed
    border =  0        # shave boader to calculate PSNR and SSIM
    model_path = os.path.join(args.model_pool, 'model_epoch_100.pth')

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(args.testsets, args.testset_name) # L_path, for Low-quality images
    H_path = L_path                               # H_path, for High-quality images
    E_path = os.path.join('results', result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    if H_path == L_path:
        args.need_degradation = True
    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    need_H = True if H_path is not None else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------
    # model = utils.get_arch(opt)
    # model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')
    # use this if BN is not merged by utils_bnorm.merge_bn(model)
    # from models.select_model import define_Model
    # model = define_Model(opt)
    # model.load_state_dict(torch.load(model_path), strict=True)
    model = utils.get_arch(args)
    utils.load_checkpoint(model, args.weights)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['avgpsnr'] = []
    test_results['avgssim'] = []

    # logger.info('model_name:{}, image sigma:{}'.format(args.model_name, args.noise_level_img))
    # logger.info(L_path)
    # L_paths = util.get_image_paths(L_path)
    # H_paths = util.get_image_paths(H_path) if need_H else None
    # # from data.readmat import loadtestall
    from dataset.mydataset import loaddata,loadtest,loadtestsentile
    x, y = loadtest()
    for idx in range(len(x)):
        img_L = x[idx,:,:,:]
        img_org = np.squeeze(img_L)
        img_H = y[idx,:,:,:]
        img_H = np.squeeze(img_H)
        img_L = torch.from_numpy(img_L)
        img_L = torch.unsqueeze(img_L,0)
        img_L = img_L.cuda()
        if True:
            img_E,S1,S2 = model(img_L)
        # # img_E = util.tensor2uint(img_E)
        # img_E = img_H
        img_E = img_E.cpu().numpy()
        img_E = np.squeeze(img_E)
        picture = img_E
        img_net = picture
        # picture = (255*picture/(np.max(img_net))).astype('uint8')
        picture = picture.astype('double')
        from PIL import Image
        pp = Image.fromarray(picture)
        name = r'G:\ffp\ExperimentPaper\FDNet\Ex1\result\FDUformer' + '/' + str(idx) + '.tif'
        pp.save(name)
        import cv2
        # import matplotlib.pyplot as plt
        # plt.imshow(picture,cmap='gray')
        # plt.show()
        max1 = np.max(img_H)
        # picture1 = (255 * img_H/(max1)).astype('uint8')
        # pp1 = Image.fromarray(picture1)
        # name = r'G:\ffp\result\denosing\tfd/uformer/test1/label' + '/' + str(idx) + '.png'
        # pp1.save(name)
        # picture1 = (255 * img_net/(max1)).astype('uint8')
        # pp1 = Image.fromarray(picture1)
        # name = r'G:\ffp\result\denosing\tfd/uformer/test1/net' + '/' + str(idx) + '.png'
        # pp1.save(name)

        # picture1 = (255 * img_org / (np.max(img_org))).astype('uint8')
        # pp1 = Image.fromarray(picture1)
        # name = r'G:\ffp\result\denosing\tfd/uformer/test1/data' + '/' + str(idx) + '.png'
        # pp1.save(name)

        img_net = np.uint8((img_net/(max1) * 255.0).round())
        img_H = np.uint8((img_H/(max1) * 255.0).round())
        img_org = np.uint8((img_org / (np.max(img_org)) * 255.0).round())
        psnr = util.calculate_psnr(img_net, img_H, border=border)
        ssim = util.calculate_ssim(img_net, img_H, border=border)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        # if(idx>100):
        #     break
    if True:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        test_results['avgpsnr'].append(ave_psnr)
        test_results['avgssim'].append(ave_ssim)
    import scipy.io as sio
    name2 = r'G:\ffp\ExperimentPaper\FDNet\Ex1\result' + '/' + 'FDUformer' + '.mat'
    sio.savemat(name2,test_results)

if __name__ == '__main__':

    main()
