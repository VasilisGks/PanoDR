import argparse
import os
import sys

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default='D:/Data/Structure3D/Structure3D/Structured3D', help="Path to the training file containing the train set files paths")
    parser.add_argument("--test_path", type=str, default='D:/Data/Structure3D/Structure3D/Structured3D_test', help="Path to the testing file containing the test set file paths")
    parser.add_argument('--results_path', type=str, default='D:/codes/DRext_files/DRext_files')
    parser.add_argument('--gt_results_path', type=str, default='D:/codes/DRext_files/DRext_files/gt/')
    parser.add_argument('--pred_results_path', type=str, default='D:/codes/DRext_files/DRext_files/pred/')
    parser.add_argument('--segmentation_model_chkpnt', type = str, default = 'D:/codes/DRext_files/Unet_layout.model', help = 'Save checkpoints here') 
    parser.add_argument('--structure_model', type=str, default="unet", choices=["unet" , "horizonnet"])
    parser.add_argument('--type_sp', type=str, default='SEAN', choices=["SEAN"])
    parser.add_argument('--model_folder', type = str, default = 'D:/codes/DRext_files/DRext_files/', help = 'Save checkpoints here')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--use_sean', type=bool, default=True) 
    parser.add_argument('--use_argmax', type=bool, default=False) 
    parser.add_argument('--lr', type=float, default=2e-4) 
    parser.add_argument('--lr_D', type=float, default=2e-4) 
    parser.add_argument('--lr_gamma', type=float, default=0.5) 
    parser.add_argument('--milestone_1', type=int, default=50) 
    parser.add_argument('--milestone_2', type=int, default=80) 
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--save_model_every', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--n_threads', type=int, default=2)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--lambda_perceptual', type=float, default=0.15) 
    parser.add_argument('--lambda_style', type=float, default = 40.0) 
    parser.add_argument('--lambda_style_patch', type=float, default = 25.0) 
    parser.add_argument('--lambda_l1', type=float, default = 4.0)  
    parser.add_argument('--lambda_l1_coarse', type=float, default = 3.0)
    parser.add_argument('--lambda_adv', type=float, default = 0.2)
    parser.add_argument('--lambda_d_match', type=float, default = 20.0) 
    parser.add_argument('--lambda_tv', type=float, default = 1.0)  #
    parser.add_argument('--viz_images_period', type=int, default=50)
    parser.add_argument('--viz_loss_period', type=int, default=10)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--adv_loss_type', type=str, default ='HingeGAN', choices=["RelativisticAverageHingeGAN", "HingeGAN", "LSGAN"]) 

    parser.add_argument('--D_max_iters', type=int, default=1)
    parser.add_argument('--pretrain_network', type=int, default=0, help = 'Model is pretrained')
    #GatedConv params
    parser.add_argument('--load_name', type = str, default ='', help = 'load model name')
    parser.add_argument('--phase', type = str, default = 'train', help = 'load model name')
    parser.add_argument('--init_type', type = str, default = 'normal', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'Adam: weight decay')
    parser.add_argument('--lr_gated', type = float, default = 0.0001, help = 'Adam: weight decay')
    parser.add_argument('--in_channels', type = int, default = 4, help = '')
    parser.add_argument('--in_layout_channels', type = int, default = 3, help = '')
    parser.add_argument('--in_SEAN_channels', type = int, default = 3, help = '')
    parser.add_argument('--style_code_dim', type = int, default = 512, help = '')
    parser.add_argument('--in_spade_channels', type = int, default = 3, help = '')
    parser.add_argument('--in_d_channels', type = int, default = 4, help = '')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'output 2D Coords')
    parser.add_argument('--latent_channels', type = int, default = 64, help = 'latent channels')
    parser.add_argument('--pad_type', type = str, default = 'spherical', help = 'the padding type', choices=["replicate", "reflection", "spherical", "zero"]) 
    parser.add_argument('--activation', type = str, default = 'relu', help = 'the activation type')
    parser.add_argument('--activation_decoder', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'in', help = 'normalization type')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: beta 1')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: beta 2')
    #visdom
    parser.add_argument("--seed", type=int, default=1337, help="Fixed manual seed, zero means no seeding.")
    parser.add_argument('-g','--gpu', type=str, default='0', help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')
    parser.add_argument('-n','--name', type=str, default='PanoDR_Inpainting_00', help='The name of this train/test. Used when storing information.') 
    parser.add_argument("--visdom", type=str, nargs='?', default='localhost', const='localhost', help = "Visdom server IP (port defaults to 8097)")
    parser.add_argument('-did','--display_id', type=int, default=1, help='')
    parser.add_argument('-d','--disp_iters', type=int, default=5, help='Log training progress (i.e. loss etc.) on console every <disp_iters> iterations.')
    parser.add_argument("--viz_loss_every", type=int, default=5, help = "Iteration interval that losses will be reported at the visdom server for visualization.")
    parser.add_argument("--viz_img_every", type=int, default=15, help = "Iteration interval that images will be reported at the visdom server for visualization.")
    #Unet discriminator 
    parser.add_argument('--num_res_blocks', type=int, default=6, help='number of residual blocks in G and D')
    parser.add_argument('--unet_discr', type=bool, default=True, help='Whether using PatchGAN or UNet Discriminator')
    parser.add_argument('--no_spectral_norm', action='store_true', help='this option deactivates spectral norm in all layers')
    parser.add_argument('--no_labelmix', action='store_true', default=False, help='if specified, do *not* use LabelMix')
    parser.add_argument('--contain_dontcare_label', default=True, help='if specified, do *not* use LabelMix')
    parser.add_argument('--no_balancing_inloss', action='store_true', default=False, help='if specified, do *not* use class balancing in the loss function')
    
    
    arguments = parser.parse_args()
    return arguments

args = parseArguments()

import torch
from torch.random import seed
from vcl3datlantis.dataloaders.Structured_3D_refined import DRS3D
from torch.utils.data import DataLoader
from vcl3datlantis.models.trainer import *
import parser
import glob
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

if __name__ == "__main__":
        device = torch.device("cuda:" + str(args.gpu_id) if (torch.cuda.is_available() and int(args.gpu_id) >= 0) else "cpu")  

        train_dataset = DataLoader(DRS3D(args.train_path,args.width, args.height, 0.8, 0.06, roll = True,  layout_extras = False), 
                args.batch_size, shuffle=True, num_workers=2)
        test_dataset =DataLoader(DRS3D(args.test_path, args.width, args.height, 0.8, 0.01, roll = False,  layout_extras = False), 
                args.test_batch_size, shuffle=False, num_workers=2)

        training(args, train_dataset, test_dataset, device)