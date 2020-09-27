import argparse
import time
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=145, help="epoch to load the saved checkpoint")
parser.add_argument("--dataset_name", type=str, default="fiveK", help="name of the dataset")
parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
parser.add_argument("--model_dir", type=str, default="LUTs/paired/fiveK_480p_3LUT_sm_1e-4_mn_10", help="directory of saved models")
opt = parser.parse_args()
opt.model_dir = opt.model_dir + '_' + opt.input_color_space

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
LUTs = torch.load("saved_models/%s/LUTs_%d.pth" % (opt.model_dir, opt.epoch))
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
classifier.load_state_dict(torch.load("saved_models/%s/classifier_%d.pth" % (opt.model_dir, opt.epoch)))
classifier.eval()

if opt.input_color_space == 'sRGB':
    dataloader = DataLoader(
        ImageDataset_sRGB("../data/%s" % opt.dataset_name,  mode="test"),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
elif opt.input_color_space == 'XYZ':
    dataloader = DataLoader(
        ImageDataset_XYZ("../data/%s" % opt.dataset_name,  mode="test"),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )


def generator(img):

    pred = classifier(img).squeeze()
    
    LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT #+ pred[3] * LUT3.LUT + pred[4] * LUT4.LUT

    combine_A = img.new(img.size())
    combine_A = trilinear_(LUT,img)

    return combine_A


def visualize_result():
    """Saves a generated sample from the validation set"""
    out_dir = "images/%s_%d" % (opt.model_dir, opt.epoch)
    os.makedirs(out_dir, exist_ok=True)
    for i, batch in enumerate(dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        img_name = batch["input_name"]
        fake_B = generator(real_A)

        #real_B = Variable(batch["A_exptC"].type(Tensor))
        #img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -1)
        #save_image(img_sample, "images/LUTs/paired/JPGsRGB8_to_JPGsRGB8_WB_original_5LUT/%s.png" % (img_name[0][:-4]), nrow=3, normalize=False)
        save_image(fake_B, os.path.join(out_dir,"%s.png" % (img_name[0][:-4])), nrow=1, normalize=False)

def test_speed():
    t_list = []
    for i in range(1,10):
        img_input = Image.open(os.path.join("../data/fiveK/input/JPG","original","a000%d.jpg"%i))
        img_input = torch.unsqueeze(TF.to_tensor(TF.resize(img_input,(4000,6000))),0)
        real_A = Variable(img_input.type(Tensor))
        torch.cuda.synchronize()
        t0 = time.time()
        for j in range(0,100):
            fake_B = generator(real_A)
        
        torch.cuda.synchronize()
        t1 = time.time()
        t_list.append(t1 - t0)
        print((t1 - t0))
    print(t_list)

# ----------
#  evaluation
# ----------
visualize_result()

#test_speed()
