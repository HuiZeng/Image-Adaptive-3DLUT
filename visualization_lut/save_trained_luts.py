import argparse
import torch

from models import *
from datasets import *


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=234, help="epoch to start training from")
parser.add_argument("--model_dir", type=str, default="LUTs/paired/fiveK_480p_3LUT_sm_1e-4_mn_10", help="path to save model")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False

criterion_pixelwise = torch.nn.MSELoss()
# Initialize generator and discriminator
LUT1 = Generator3DLUT_identity()
LUT2 = Generator3DLUT_zero()
LUT3 = Generator3DLUT_zero()
#LUT4 = Generator3DLUT_2()
#LUT5 = Generator3DLUT_2()


# Load pretrained models
LUTs = torch.load("saved_models/%s/LUTs_%d.pth" % (opt.model_dir, opt.epoch))
LUT1.load_state_dict(LUTs["1"])
LUT2.load_state_dict(LUTs["2"])
LUT3.load_state_dict(LUTs["3"])
#LUT4.load_state_dict(LUTs["4"])
#LUT5.load_state_dict(LUTs["5"])
LUT1.eval()
LUT2.eval()
LUT3.eval()
#LUT4.eval()
#LUT5.eval()


f = open('visualization/learned_LUT_%d_1.txt'%opt.epoch,'a')
for p in range(0,LUT1.LUT.shape[0]):
    for i in range(0,LUT1.LUT.shape[1]):
        for j in range(0,LUT1.LUT.shape[2]):
            for k in range(0,LUT1.LUT.shape[3]):
                f.write("%f\n"%LUT1.LUT[p,i,j,k].detach().numpy())
f.close()
f = open('visualization/learned_LUT_%d_2.txt'%opt.epoch,'a')
for p in range(0,LUT2.LUT.shape[0]):
    for i in range(0,LUT2.LUT.shape[1]):
        for j in range(0,LUT2.LUT.shape[2]):
            for k in range(0,LUT2.LUT.shape[3]):
                f.write("%f\n"%LUT2.LUT[p,i,j,k].detach().numpy())
f.close()
f = open('visualization/learned_LUT_%d_3.txt'%opt.epoch,'a')
for p in range(0,LUT3.LUT.shape[0]):
    for i in range(0,LUT3.LUT.shape[1]):
        for j in range(0,LUT3.LUT.shape[2]):
            for k in range(0,LUT3.LUT.shape[3]):
                f.write("%f\n"%LUT3.LUT[p,i,j,k].detach().numpy())
f.close()

