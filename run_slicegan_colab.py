### Welcome to SliceGAN ###
####### Steve Kench #######
'''
Use this file to define your settings for a training run, or
to generate a synthetic image using a trained generator.
'''

from matplotlib import use
from slicegan import model, networks, util, Circularity
import argparse

# Run with False to show an image during or after training
parser = argparse.ArgumentParser()

# 0 Evaluation
# 1 Training
parser.add_argument('training', type=int)

# 0 for no CircNet
# 1 for CircNet WITHOUT training
# 2 for CircNet WITH training
parser.add_argument("use_Circ", type=int)
parser.add_argument("noise_type", type=str)

args = parser.parse_args()
Training = args.training
use_Circ = args.use_Circ
noise_type = args.noise_type

# Define project name
Project_name = f'{noise_type}_noise'
# Specify project folder.
Project_dir = 'drive/MyDrive/Deep Learning/sliceGAN/Coding/Trained_Generators'

Project_path = util.mkdr(Project_name, Project_dir, Training)

## Data Processing
# Define image  type (colour, grayscale, three-phase or two-phase.
# n-phase materials must be segmented)
image_type = 'twophase'
# define data type (for colour/grayscale images, must be 'colour' / '
# greyscale. nphase can be, 'tif', 'png', 'jpg','array')
data_type = 'tif'
# Path to your data. One string for isotrpic, 3 for anisotropic
data_path = ['drive/MyDrive/Deep Learning/sliceGAN/Coding/data/3D_data_bin_processed.tif']

## Network Architectures
# Training image size, no. channels and scale factor vs raw data
img_size, img_channels, scale_factor = 64, 2,  1
# z vector depth
z_channels = 16
# Layers in G and D
lays = 6

# Type of noise distribution
# noise_type = "normal"
# kernals for each layer
# dk, gk = [4]*lays, [4]*lays
# # strides
# ds, gs = [2]*lays, [2]*lays
# # no. filters
# df, gf = [img_channels,64,128,256,512,1], [z_channels,512,256,128,64,img_channels]
# # paddings
# dp, gp = [1,1,1,1,0],[2,2,2,2,3]

net_params = {

    "pth": Project_path,
    "Training": Training,
    "imtype": 'threephase',

    "dk" : [4]*lays,
    "gk" : [4]*lays,

    "ds": [2]*lays,
    "gs": [2]*lays,

    "df": [img_channels,64,128,256,512,1],
    "gf": [z_channels,512,256,128,64,img_channels],

    "dp": [1,1,1,1,0],
    "gp": [2,2,2,2,3],
    }

## Create Networks

# Test efficacy of Blob Detector on Real Image

# imm = util.testCircleDetector(data_path)
# ts = Circularity.numCircles(imm)

if use_Circ != 0:    ## Create and Train CircleNet
    ## Create Path
    
    Circle_dir = 'TrainedCNet'
    W_dir = 'weights'
    img_dir = 'img'

    Circle_path = util.mkdr(Project_name, Circle_dir, W_dir)
    # blob_path = util.mkdr(Project_name, Circle_dir, img_dir)
    # Test efficacy of Blob Detector on Real Image

    # imm = util.testCircleDetector(data_path, blob_path)
    # ts = Circularity.numCircles(imm)

    ## Create and Train CircleNet

    circleNet = Circularity.init_circleNet(net_params["dk"], net_params["ds"], net_params["df"], net_params["dp"])
    
    if use_Circ == 1:
        circleNet = Circularity.CircleWeights(circleNet, Circle_path, False)

    if use_Circ == 2:
        Circularity.trainCNet(data_type, data_path, img_size, scale_factor, circleNet)
        Circularity.CircleWeights(circleNet, Circle_path, True)

    

## Create GAN
netD, netG = networks.slicegan_nets(**net_params)
# netD, netG = networks.slicegan_nets(Project_path, Training, image_type, dk, ds, df, dp, gk, gs, gf, gp)

lz_calced = model.lz_img_size_converter(net_params["gk"], net_params["gs"], net_params["gp"], img_size)

# Train
if Training:
    train_params = {
        "pth": Project_path,
        "imtype": image_type,
        "datatype": data_type,
        "real_data": data_path,
        "Disc": netD,
        "Gen": netG,
        "nc": img_channels,
        "l": img_size,
        "nz": z_channels,
        "sf": scale_factor,
        "lz": lz_calced,
        "num_epochs": 10,
        "use_Circ": use_Circ,
        "noise_type": noise_type
    }

    model.train(**train_params)
else:
    test_params = {
        "pth": Project_path,
        "imtype": image_type,
        "Gen": netG(),
        "nz": z_channels,
        "lf": 4,
        "periodic": False,
        "noise_type": noise_type
    }
    img, raw, netG = util.test_img(**test_params)