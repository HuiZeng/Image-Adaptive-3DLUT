import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import numpy as np
import math
from trilinear_c._ext import trilinear

def weights_init_normal_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class resnet18_224(nn.Module):

    def __init__(self, out_dim=5, aug_test=False):
        super(resnet18_224, self).__init__()

        self.aug_test = aug_test
        net = models.resnet18(pretrained=True)
        # self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        # self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

        self.upsample = nn.Upsample(size=(224,224),mode='bilinear')
        net.fc = nn.Linear(512, out_dim)
        self.model = net


    def forward(self, x):

        x = self.upsample(x)
        if self.aug_test:
            # x = torch.cat((x, torch.rot90(x, 1, [2, 3]), torch.rot90(x, 3, [2, 3])), 0)
            x = torch.cat((x, torch.flip(x, [3])), 0)
        f = self.model(x)

        return f

##############################
#           DPE
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 5, 2, 2)]
        layers.append(nn.SELU(inplace=True))
        if normalize:
            #layers.append(nn.BatchNorm2d(out_size))
            nn.InstanceNorm2d(out_size, affine = True)
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True),
            nn.Conv2d(in_size, out_size, 3, padding=1),
            nn.SELU(inplace=True),
        ]

        if normalize:
            #layers.append(nn.BatchNorm2d(out_size))
            nn.InstanceNorm2d(out_size, affine = True)

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.SELU(inplace=True),
            #nn.BatchNorm2d(16),
            nn.InstanceNorm2d(16, affine = True),
            )
        self.down1 = UNetDown(16, 32)
        self.down2 = UNetDown(32, 64)
        self.down3 = UNetDown(64, 128)
        self.down4 = UNetDown(128, 128)
        self.down5 = UNetDown(128, 128)
        self.down6 = UNetDown(128, 128)
        self.down7 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(128, 128, 1, padding=0),
            )

        self.upsample = nn.Upsample(scale_factor=4, mode = 'bilinear')
        self.conv1x1 = nn.Conv2d(256, 128, 1, padding=0)

        self.up1 = UNetUp(128, 128)
        self.up2 = UNetUp(256, 128)
        self.up3 = UNetUp(192, 64)
        self.up4 = UNetUp(96, 32)

        self.final = nn.Sequential(
            nn.Conv2d(48, 16, 3, padding=1),
            nn.SELU(inplace=True),
            #nn.BatchNorm2d(16),
            #nn.InstanceNorm2d(16, affine = True),
            nn.Conv2d(16, out_channels, 3, padding=1),
            #nn.Tanh(),
            )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder

        x1 = self.conv1(x)
        d1 = self.down1(x1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        d8 = self.upsample(d7)
        d9 = torch.cat((d4, d8), 1)
        d9 = self.conv1x1(d9)

        u1 = self.up1(d9, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        u4 = self.up4(u3, x1)

        return torch.add(self.final(u4), x)


class Discriminator_UNet(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator_UNet, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 128),
            *discriminator_block(128, 128),
            nn.Conv2d(128, 1, 4, padding=0)
        )

    def forward(self, img_input):
        return self.model(img_input)

##############################
#        Discriminator
##############################


def discriminator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        #layers.append(nn.BatchNorm2d(out_filters))

    return layers

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256,256),mode='bilinear'),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 128),
            #*discriminator_block(128, 128),
            nn.Conv2d(128, 1, 8, padding=0)
        )

    def forward(self, img_input):
        return self.model(img_input)

class Classifier(nn.Module):
    def __init__(self, in_channels=3):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256,256),mode='bilinear'),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32, normalization=True),
            *discriminator_block(32, 64, normalization=True),
            *discriminator_block(64, 128, normalization=True),
            *discriminator_block(128, 128),
            #*discriminator_block(128, 128, normalization=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 3, 8, padding=0),
        )

    def forward(self, img_input):
        return self.model(img_input)

class Classifier_unpaired(nn.Module):
    def __init__(self, in_channels=3):
        super(Classifier_unpaired, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256,256),mode='bilinear'),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 128),
            #*discriminator_block(128, 128),
            nn.Conv2d(128, 3, 8, padding=0),
        )

    def forward(self, img_input):
        return self.model(img_input)


class Generator3DLUT_identity(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_identity, self).__init__()
        if dim == 33:
            file = open("IdentityLUT33.txt",'r')
        elif dim == 64:
            file = open("IdentityLUT64.txt",'r')
        LUT = file.readlines()
        self.LUT = torch.zeros(3,dim,dim,dim, dtype=torch.float)

        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    n = i * dim*dim + j * dim + k
                    x = LUT[n].split()
                    self.LUT[0,i,j,k] = float(x[0])
                    self.LUT[1,i,j,k] = float(x[1])
                    self.LUT[2,i,j,k] = float(x[2])
        self.LUT = nn.Parameter(torch.tensor(self.LUT))
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):

        return self.TrilinearInterpolation(self.LUT, x)


class Generator3DLUT_zero(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_zero, self).__init__()

        self.LUT = torch.zeros(3,dim,dim,dim, dtype=torch.float)
        self.LUT = nn.Parameter(torch.tensor(self.LUT))
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):

        return self.TrilinearInterpolation(self.LUT, x)

class TrilinearInterpolation(torch.autograd.Function):

    def forward(self, LUT, x):

        x = x.contiguous()
        output = x.new(x.size())
        dim = LUT.size()[-1]
        shift = dim ** 3
        binsize = 1.0001 / (dim-1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)

        self.x = x
        self.LUT = LUT
        self.dim = dim
        self.shift = shift
        self.binsize = binsize
        self.W = W
        self.H = H
        self.batch = batch

        if x.is_cuda:
            if batch == 1:
                trilinear.trilinear_forward_cuda(LUT,x,output,dim,shift,binsize,W,H,batch)
            elif batch > 1:
                output = output.permute(1,0,2,3).contiguous()
                trilinear.trilinear_forward_cuda(LUT,x.permute(1,0,2,3).contiguous(),output,dim,shift,binsize,W,H,batch)
                output = output.permute(1,0,2,3).contiguous()

        else:
            trilinear.trilinear_forward(LUT,x,output,dim,shift,binsize,W,H,batch)

        return output

    def backward(self, grad_x):

        grad_LUT = torch.zeros(3,self.dim,self.dim,self.dim,dtype=torch.float)

        if grad_x.is_cuda:
            grad_LUT = grad_LUT.cuda()
            if self.batch == 1:
                trilinear.trilinear_backward_cuda(self.x,grad_x,grad_LUT,self.dim,self.shift,self.binsize,self.W,self.H,self.batch)
            elif self.batch > 1:
                trilinear.trilinear_backward_cuda(self.x.permute(1,0,2,3).contiguous(),grad_x.permute(1,0,2,3).contiguous(),grad_LUT,self.dim,self.shift,self.binsize,self.W,self.H,self.batch)
        else:
            trilinear.trilinear_backward(self.x,grad_x,grad_LUT,self.dim,self.shift,self.binsize,self.W,self.H,self.batch)

        return grad_LUT, None


class TV_3D(nn.Module):
    def __init__(self, dim=33):
        super(TV_3D,self).__init__()

        self.weight_r = torch.ones(3,dim,dim,dim-1, dtype=torch.float)
        self.weight_r[:,:,:,(0,dim-2)] *= 2.0
        self.weight_g = torch.ones(3,dim,dim-1,dim, dtype=torch.float)
        self.weight_g[:,:,(0,dim-2),:] *= 2.0
        self.weight_b = torch.ones(3,dim-1,dim,dim, dtype=torch.float)
        self.weight_b[:,(0,dim-2),:,:] *= 2.0
        self.relu = torch.nn.ReLU()

    def forward(self, LUT):

        dif_r = LUT.LUT[:,:,:,:-1] - LUT.LUT[:,:,:,1:]
        dif_g = LUT.LUT[:,:,:-1,:] - LUT.LUT[:,:,1:,:]
        dif_b = LUT.LUT[:,:-1,:,:] - LUT.LUT[:,1:,:,:]
        tv = torch.mean(torch.mul((dif_r ** 2),self.weight_r)) + torch.mean(torch.mul((dif_g ** 2),self.weight_g)) + torch.mean(torch.mul((dif_b ** 2),self.weight_b))

        mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))

        return tv, mn


