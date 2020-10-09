import argparse
import torch
import os
import numpy as np
import cv2
from PIL import Image

from models import *
import torchvision_x_functional as TF_x
import torchvision.transforms.functional as TF


parser = argparse.ArgumentParser()

parser.add_argument("--image_dir", type=str, default="demo_images", help="directory of image")
parser.add_argument("--image_name", type=str, default="a1629.jpg", help="name of image")
parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
parser.add_argument("--model_dir", type=str, default="pretrained_models", help="directory of pretrained models")
parser.add_argument("--output_dir", type=str, default="demo_results", help="directory to save results")
opt = parser.parse_args()
opt.model_dir = opt.model_dir + '/' + opt.input_color_space
opt.image_path = opt.image_dir + '/' + opt.input_color_space + '/' + opt.image_name
os.makedirs(opt.output_dir, exist_ok=True)

# use gpu when detect cuda
cuda = True if torch.cuda.is_available() else False
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

criterion_pixelwise = torch.nn.MSELoss()
LUT0 = Generator3DLUT_identity()
LUT1 = Generator3DLUT_zero()
LUT2 = Generator3DLUT_zero()
#LUT3 = Generator3DLUT_zero()
#LUT4 = Generator3DLUT_zero()
classifier = Classifier()
trilinear_ = TrilinearInterpolation() 

if cuda:
    LUT0 = LUT0.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    #LUT3 = LUT3.cuda()
    #LUT4 = LUT4.cuda()
    classifier = classifier.cuda()
    criterion_pixelwise.cuda()

# Load pretrained models
LUTs = torch.load("%s/LUTs.pth" % opt.model_dir)
LUT0.load_state_dict(LUTs["0"])
LUT1.load_state_dict(LUTs["1"])
LUT2.load_state_dict(LUTs["2"])
#LUT3.load_state_dict(LUTs["3"])
#LUT4.load_state_dict(LUTs["4"])
LUT0.eval()
LUT1.eval()
LUT2.eval()
#LUT3.eval()
#LUT4.eval()
classifier.load_state_dict(torch.load("%s/classifier.pth" % opt.model_dir))
classifier.eval()


def generate_LUT(img):

    pred = classifier(img).squeeze()
    
    LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT #+ pred[3] * LUT3.LUT + pred[4] * LUT4.LUT

    return LUT


# ----------
#  test
# ----------
# read image and transform to tensor
if opt.input_color_space == 'sRGB':
    img = Image.open(opt.image_path)
    img = TF.to_tensor(img).type(Tensor)
elif opt.input_color_space == 'XYZ':
    img = cv2.imread(opt.image_path, -1)
    img = np.array(img)
    img = TF_x.to_tensor(img).type(Tensor)
img = img.unsqueeze(0)

LUT = generate_LUT(img)

# generate image
result = trilinear_(LUT, img)

# save image
ndarr = result.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
im = Image.fromarray(ndarr)
im.save('%s/result.jpg' % opt.output_dir, quality=95)


