import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd

from models_x import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from, 0 starts from scratch, >0 starts from saved checkpoints")
parser.add_argument("--n_epochs", type=int, default=800, help="total number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="fiveK", help="name of the dataset")
parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lambda_pixel", type=float, default=1000, help="content preservation weight: 1000 for sRGB input, 10 for XYZ input")
parser.add_argument("--lambda_gp", type=float, default=10, help="gradient penalty weight in wgan-gp")
parser.add_argument("--lambda_smooth", type=float, default=1e-4, help="smooth regularization")
parser.add_argument("--lambda_monotonicity", type=float, default=10.0, help="monotonicity regularization: 10 for sRGB input, 100 for XYZ input (slightly better)")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
parser.add_argument("--output_dir", type=str, default="LUTs/unpaired/fiveK_480p_sm_1e-4_mn_10_pixel_1000", help="path to save model")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
opt = parser.parse_args()
opt.output_dir = opt.output_dir + '_' + opt.input_color_space
print(opt)

os.makedirs("saved_models/%s" % opt.output_dir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.MSELoss()

# Initialize generator and discriminator
LUT0 = Generator3DLUT_identity()
LUT1 = Generator3DLUT_zero()
LUT2 = Generator3DLUT_zero()
#LUT3 = Generator3DLUT_zero()
#LUT4 = Generator3DLUT_zero()
classifier = Classifier_unpaired()
discriminator = Discriminator()
TV3 = TV_3D()

if cuda:
    LUT0 = LUT0.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    #LUT3 = LUT3.cuda()
    #LUT4 = LUT4.cuda()
    classifier = classifier.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
    discriminator = discriminator.cuda()
    TV3.cuda()
    TV3.weight_r = TV3.weight_r.type(Tensor)
    TV3.weight_g = TV3.weight_g.type(Tensor)
    TV3.weight_b = TV3.weight_b.type(Tensor)

if opt.epoch != 0:
    # Load pretrained models
    LUTs = torch.load("saved_models/%s/LUTs_%d.pth" % (opt.output_dir, opt.epoch))
    LUT0.load_state_dict(LUTs["0"])
    LUT1.load_state_dict(LUTs["1"])
    LUT2.load_state_dict(LUTs["2"])
    #LUT3.load_state_dict(LUTs["3"])
    #LUT4.load_state_dict(LUTs["4"])
    classifier.load_state_dict(torch.load("saved_models/%s/classifier_%d.pth" % (opt.output_dir, opt.epoch)))
else:
    # Initialize weights
    classifier.apply(weights_init_normal_classifier)
    torch.nn.init.constant_(classifier.model[12].bias.data, 1.0)
    discriminator.apply(weights_init_normal_classifier)

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(classifier.parameters(), LUT0.parameters(),LUT1.parameters(),LUT2.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)) #,LUT3.parameters(),LUT4.parameters()
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

if opt.input_color_space == 'sRGB':
    dataloader = DataLoader(
        ImageDataset_sRGB_unpaired("../data/%s" % opt.dataset_name, mode="train"),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    psnr_dataloader = DataLoader(
        ImageDataset_sRGB_unpaired("../data/%s" % opt.dataset_name,  mode="test"),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
elif opt.input_color_space == 'XYZ':
    dataloader = DataLoader(
        ImageDataset_XYZ_unpaired("../data/%s" % opt.dataset_name, mode="train"),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    psnr_dataloader = DataLoader(
        ImageDataset_XYZ_unpaired("../data/%s" % opt.dataset_name,  mode="test"),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

def calculate_psnr():
    classifier.eval()
    avg_psnr = 0
    for i, batch in enumerate(psnr_dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))
        fake_B, weights_norm = generator(real_A)
        fake_B = torch.round(fake_B*255)
        real_B = torch.round(real_B*255)
        mse = criterion_pixelwise(fake_B, real_B)
        psnr = 10 * math.log10(255.0 * 255.0 / mse.item())
        avg_psnr += psnr

    return avg_psnr/ len(psnr_dataloader)


def visualize_result(epoch):
    """Saves a generated sample from the validation set"""
    os.makedirs("images/LUTs/" +str(epoch), exist_ok=True)
    for i, batch in enumerate(psnr_dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))
        img_name = batch["input_name"]
        fake_B, weights_norm = generator(real_A)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -1)
        fake_B = torch.round(fake_B*255)
        real_B = torch.round(real_B*255)
        mse = criterion_pixelwise(fake_B, real_B)
        psnr = 10 * math.log10(255.0 * 255.0 / mse.item())
        save_image(img_sample, "images/LUTs/%s/%s.jpg" % (epoch, img_name[0]+'_'+str(psnr)[:5]), nrow=3, normalize=False)


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def generator(img):

    pred = classifier(img).squeeze()
    weights_norm = torch.mean(pred ** 2)
    combine_A = pred[0] * LUT0(img) + pred[1] * LUT1(img) + pred[2] * LUT2(img) #+ pred[3] * LUT3(img) + pred[4] * LUT4(img)

    return combine_A, weights_norm

# ----------
#  Training
# ----------

avg_psnr = calculate_psnr()
print(avg_psnr)
prev_time = time.time()
max_psnr = 0
max_epoch = 0
for epoch in range(opt.epoch, opt.n_epochs):
    loss_D_avg = 0
    loss_G_avg = 0
    loss_pixel_avg = 0
    cnt = 0
    psnr_avg = 0
    classifier.train()
    for i, batch in enumerate(dataloader):

        # Model inputs
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["B_exptC"].type(Tensor))


        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        fake_B, weights_norm = generator(real_A)
        pred_real = discriminator(real_B)
        pred_fake = discriminator(fake_B)

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_B, fake_B)

        # Total loss
        loss_D = -torch.mean(pred_real) + torch.mean(pred_fake) + opt.lambda_gp * gradient_penalty

        loss_D.backward()
        optimizer_D.step()

        loss_D_avg += (-torch.mean(pred_real) + torch.mean(pred_fake)) / 2

        # ------------------
        #  Train Generators
        # ------------------
        if i % opt.n_critic == 0:

            optimizer_G.zero_grad()

            fake_B, weights_norm = generator(real_A)
            pred_fake = discriminator(fake_B)
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_A)

            tv0, mn0 = TV3(LUT0)
            tv1, mn1 = TV3(LUT1)
            tv2, mn2 = TV3(LUT2)
            #tv3, mn3 = TV3(LUT3)
            #tv4, mn4 = TV3(LUT4)
            
            tv_cons = tv0 + tv1 + tv2 #+ tv3 + tv4
            mn_cons = mn0 + mn1 + mn2 #+ mn3 + mn4

            loss_G = -torch.mean(pred_fake) + opt.lambda_pixel * loss_pixel + opt.lambda_smooth * (weights_norm + tv_cons) + opt.lambda_monotonicity * mn_cons

            loss_G.backward()

            optimizer_G.step()

            cnt += 1
            loss_G_avg += -torch.mean(pred_fake)

            loss_pixel_avg += loss_pixel
            psnr_avg += 10 * math.log10(1 / loss_pixel.item())


        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D: %f, G: %f] [pixel: %f] [tv: %f, wnorm: %f, mn: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D_avg.item() / cnt,
                loss_G_avg.item() / cnt,
                loss_pixel_avg.item() / cnt,
                tv_cons, weights_norm, mn_cons,
                time_left,
            )
        )

        # If at sample interval save image
    avg_psnr = calculate_psnr()
    if avg_psnr > max_psnr:
        max_psnr = avg_psnr
        max_epoch = epoch
    sys.stdout.write(" [PSNR: %f] [max PSNR: %f, epoch: %d]\n"% (avg_psnr, max_psnr, max_epoch))

    #if (epoch+1) % 10 == 0:
    #    visualize_result(epoch+1)

    if epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        LUTs = {"0": LUT0.state_dict(), "1": LUT1.state_dict(), "2": LUT2.state_dict()} #, "3": LUT3.state_dict(), "4": LUT4.state_dict()
        torch.save(LUTs, "saved_models/%s/LUTs_%d.pth" % (opt.output_dir, epoch))
        torch.save(classifier.state_dict(), "saved_models/%s/classifier_%d.pth" % (opt.output_dir, epoch))
        file = open('saved_models/%s/result.txt' % opt.output_dir,'a')
        file.write(" [PSNR: %f] [max PSNR: %f, epoch: %d]\n"% (avg_psnr, max_psnr, max_epoch))
        file.close()
