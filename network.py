from __future__ import print_function
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
import os


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias)


def conv1x1(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1
    )

def conv1(in_channels, out_channels, bias=True):
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=1
    )


class DsBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pooling):
        super(DsBlock, self).__init__()
        self.conv = conv3x3(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pooling = pooling
        if pooling:
            self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):

        out = self.conv(x)
        out = self.relu(out)
        before_pool = out
        if self.pooling:
            out = self.mp(out)

        return out, before_pool

class DsBlockNoSkip(nn.Module):

    def __init__(self, in_channels, out_channels, pooling):
        super(DsBlockNoSkip, self).__init__()
        self.conv = conv3x3(in_channels, out_channels)
        # self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.pooling = pooling
        if pooling:
            self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.conv2 = conv3x3x3(planes, planes)
        # self.bn2 = nn.BatchNorm3d(planes)
        # self.stride = stride

    def forward(self, x):

        out = self.conv(x)
        out = self.relu(out)
        if self.pooling:
            out = self.mp(out)

        return out


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


class UsBlock(nn.Module):

    def __init__(self, in_channels, out_channels, up_mode='transpose'):
        super(UsBlock, self).__init__()

        self.upconv = upconv2x2(in_channels, out_channels, mode=up_mode)
        self.conv = conv3x3(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, before_pool, x):
        x = self.upconv(x)
        x = x + before_pool
        x = self.conv(x)
        x = self.relu(x)
        return x

class UsBlockNoSkip(nn.Module):

    def __init__(self, in_channels, out_channels, up_mode='transpose'):
        super(UsBlockNoSkip, self).__init__()

        self.upconv = upconv2x2(in_channels, out_channels, mode=up_mode)
        self.conv = conv3x3(out_channels, out_channels)

    def forward(self, x):

        x = self.upconv(x)
        x = self.conv(x)
        return x


class FCN(nn.Module):
    '''
    This class implements a FCN, WITHOUT including skipping connections. The input construction arguments: num_classes, 
    in_channels, depth (down sampling number is depth-1), start_filter number, upsampling mode.
    '''

    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filts=64, up_mode='transpose'):
        super(FCN, self).__init__()

        self.down_convs = []
        self.up_convs = []

        # put one conv  at the beginning
        self.conv_start = conv3x3(in_channels, start_filts, stride=1)
        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = start_filts * (2 ** i)
            outs = start_filts * (2 ** (i + 1)) if i < depth - \
                1 else start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DsBlockNoSkip(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UsBlockNoSkip(ins, outs, up_mode=up_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

    def forward(self, x, logSoftmax=True):
        x = self.conv_start(x)

        for i, module in enumerate(self.down_convs):
            x = module(x)

        for i, module in enumerate(self.up_convs):
            x = module(x)

        x = self.conv_final(x)
        x = x.squeeze(1)
        if logSoftmax:
            x = torch.nn.functional.log_softmax(x, dim=1)
        else:
            x = torch.nn.functional.softmax(x, dim=1)
        return x

class UNet(nn.Module):
    '''
    This class implements a Unet with global and local residual connections. The input construction arguments: num_classes, 
    in_channels, depth (down sampling number is depth-1), start_filter number, upsampling mode.
    '''

    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filts=64, up_mode='transpose', PairNet=False):
        super(UNet, self).__init__()

        self.down_convs = []
        self.up_convs = []
        self.pn = PairNet

        # put one conv  at the beginning
        self.conv_start = conv3x3(in_channels, start_filts, stride=1)
        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = start_filts * (2 ** i)
            outs = start_filts * (2 ** (i + 1))
            pooling = True if i < depth - 1 else False

            down_conv = DsBlock(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UsBlock(ins, outs, up_mode=up_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

    def forward(self, x, logSoftmax=True):
        x = self.conv_start(x)
        encoder_outs = []

        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        # print(len(encoder_outs))
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)
        if self.pn:
            return x
        x = x.squeeze(1)
        if logSoftmax:
            x = torch.nn.functional.log_softmax(x, dim=1)
        else:
            x = torch.nn.functional.softmax(x, dim=1)
        return x



class PairNet(nn.Module):
    """
    Do not support mutliple surfaces yet.
    """
    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filts=64, up_mode='transpose', col_len=512, fc_inter=128, left_nbs=3):
        super(PairNet, self).__init__()
        self.Unet = UNet(num_classes, in_channels, depth,
                 start_filts, up_mode, PairNet=True)
        self.FC_D1 = conv1(col_len*(left_nbs+1)*2, fc_inter)
        self.FC_D2 = conv1(fc_inter, 1)
        self.half = left_nbs+1

    def forward(self, x):
        x = self.Unet(x).squeeze(1)
        t_list = []
        for i in range(self.half):
            if i==0:
                t_list.append(x[:,:,:-1])
                t_list.append(x[:,:,1:])
            else:
                t_list.append(torch.cat((x[:,:,0:1].expand(-1,-1,i), x[:,:,:-(i+1)]), -1))
                t_list.append(torch.cat((x[:,:,i+1:], x[:,:,-1:].expand(-1,-1,i)), -1))
        t_list = torch.cat(t_list, 1)
        # print(t_list.size())
        D = self.FC_D1(t_list)
        D = torch.nn.functional.relu(D)
        D = self.FC_D2(D)
        D = D.squeeze(1)
        return D

# define the clapping threshold for probability normalization.
STAB_NB = 1e-15


def newton_sol_pd(g_mean, g_sigma, w_comp, d_p):
    '''
    This function solves the quadratic CRF: 1/2 xT H x + pT x. Assume g_mean has shape: bn,  x_len. w_comp is a torch parameter.
    The size of d_p is one less than g_mean, since we currently do not consider the head and tail difference.
    '''
    x_len = g_mean.size(1)
    # The Hessian is divided into two parts: pairwise and unary.
    hess_pair = torch.diag(-2.*w_comp.repeat(x_len-1), diagonal=-1) + torch.diag(-2.*w_comp.repeat(x_len-1),
                    diagonal=1) + torch.diag(torch.cat((2.*w_comp, 4.*w_comp.repeat(x_len-2), 2.*w_comp), dim=0),
                        diagonal=0)
    # hess_pair = torch.stack(hess_pair)
    # pairwise parts are the same across patches within a batch
    if g_mean.is_cuda:
        hess_pair_batch = torch.stack([hess_pair]*g_sigma.size(0)).cuda()
    else:
        hess_pair_batch = torch.stack([hess_pair]*g_sigma.size(0))
    # get reverse of sigma array
    g_sigma_rev = 1./g_sigma
    # convert sigma reverse array to diagonal matrices
    g_sigma_eye = torch.diag_embed(g_sigma_rev)
    # sum up two parts
    hess_batch = hess_pair_batch + g_sigma_eye
    # compute inverse of Hessian
    hess_inv_batch = torch.inverse(hess_batch)
    # generate the linear coefficient P
    p_u = g_mean/g_sigma
    
    # print(p_u.size(), d_p.size())
    delta = 2.*(torch.cat((d_p, torch.zeros(d_p.size(0), 1).cuda()), dim=-1) - 
                                        torch.cat((torch.zeros(d_p.size(0), 1).cuda(), d_p), dim=-1))
    p = p_u + delta
    # solve it globally
    solution = torch.matmul(hess_inv_batch, p.unsqueeze(-1)).squeeze(-1)

    return solution

def normalize_prob(x):
    '''Normalize prob map to [0, 1]. Numerically, add 1e-6 to all. Assume the last dimension is prob map.'''
    x_norm = (x - x.min(-1, keepdim=True)
              [0]) / (x.max(-1, keepdim=True)[0] - x.min(-1, keepdim=True)[0])
    x_norm += 1e-3
    return x_norm


def gaus_fit(x, tr_flag=True):
    '''This module is designed to regress Gaussian function. Weighted version is chosen. The input tensor should
    have the format: BN,  X_LEN, COL_LEN.'''
    bn,  x_len, col_len = tuple(x.size())
    col_ind_set = torch.arange(col_len).expand(
        bn,  x_len, col_len).double()
    if x.is_cuda:
        col_ind_set = col_ind_set.cuda()
    y = x.double()
    lny = torch.log(y).double()
    y2 = torch.pow(y, 2).double()
    x2 = torch.pow(col_ind_set, 2).double()
    sum_y2 = torch.sum(y2, dim=-1)
    sum_xy2 = torch.sum(col_ind_set * y2, dim=-1)
    sum_x2y2 = torch.sum(x2 * y2, dim=-1)
    sum_x3y2 = torch.sum(torch.pow(col_ind_set, 3) * y2, dim=-1)
    sum_x4y2 = torch.sum(torch.pow(col_ind_set, 4) * y2, dim=-1)
    sum_y2lny = torch.sum(y2 * lny, dim=-1)
    sum_xy2lny = torch.sum(col_ind_set * y2 * lny, dim=-1)
    sum_x2y2lny = torch.sum(x2 * y2 * lny, dim=-1)
    b_num = (sum_x2y2**2*sum_xy2lny - sum_y2*sum_x4y2*sum_xy2lny + sum_xy2*sum_x4y2*sum_y2lny +
             sum_y2*sum_x3y2*sum_x2y2lny - sum_x2y2*sum_x3y2*sum_y2lny - sum_xy2*sum_x2y2*sum_x2y2lny)
    c_num = (sum_x2y2lny*sum_xy2**2 - sum_xy2lny*sum_xy2*sum_x2y2 - sum_x3y2*sum_y2lny*sum_xy2 +
             sum_y2lny*sum_x2y2**2 - sum_y2*sum_x2y2lny*sum_x2y2 + sum_y2*sum_x3y2*sum_xy2lny)
    c_num[(c_num < STAB_NB) & (c_num > -STAB_NB)
          ] = torch.sign(c_num[(c_num < STAB_NB) & (c_num > -STAB_NB)]) * STAB_NB
    mu = -b_num / (2.*c_num)

    c_din = sum_x4y2*sum_xy2**2 - 2*sum_xy2*sum_x2y2*sum_x3y2 + \
        sum_x2y2**3 - sum_y2*sum_x4y2*sum_x2y2 + sum_y2*sum_x3y2**2
    sigma_b_sqrt = -0.5*c_din/c_num
    sigma_b_sqrt[sigma_b_sqrt < 1] = 1
    sigma = sigma_b_sqrt
    #TODO May have better strategies to handle the failure of Gaussian fitting.
    if not tr_flag:
        mu[mu >= col_len-1] = col_len-1
        mu[mu <= 0] = 0.
    if torch.isnan(mu).any() or torch.isnan(sigma).any():
        raise Exception("mu or sigma gets NaN value.")

    mu = mu.float()
    sigma = sigma.float()

    return mu, sigma

class SurfSegNet(torch.nn.Module):
    """
    ONly GPU version has been implemented!!!
    """
    def __init__(self, unary_model, hps, wt_init=1e-5,  pair_model=None):
        super(SurfSegNet, self).__init__()
        self.unary = unary_model
        self.pair = pair_model
        self.hps = hps
        self.w_comp = torch.nn.Parameter(torch.ones(1)*wt_init)
    def load_wt(self):
        if os.path.isfile(self.hps['surf_net']['resume_path']):
            print('loading surfnet checkpoint: {}'.format(self.hps['surf_net']['resume_path']))
            checkpoint = torch.load(self.hps['surf_net']['resume_path'])
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded surfnet checkpoint (epoch {})"
                .format(checkpoint['epoch']))
        else:
            if os.path.isfile(self.hps['surf_net']['unary_pretrain_path']):
                print('loading unary network pretrain checkpoint: {}'.format(self.hps['surf_net']['unary_pretrain_path']))
                checkpoint = torch.load(self.hps['surf_net']['unary_pretrain_path'])
                # stripe the unary prefix of pretrained model state dict
                if self.hps['surf_net']['unary_pretrain_sub']:
                    new_stat_dict = {}
                    for key, value in checkpoint['state_dict'].items():
                        new_stat_dict[key[6:]] = value
                    self.unary.load_state_dict(new_stat_dict)
                else:
                    self.unary.load_state_dict(checkpoint['state_dict'])
                print("=> loaded unary network pretrain checkpoint (epoch {})"
                    .format(checkpoint['epoch']))
            if self.pair is None:
                print("Zero prior is used.")
            elif os.path.isfile(self.hps['surf_net']['pair_pretrain_path']):
                print('loading pair network pretrain checkpoint: {}'.format(self.hps['surf_net']['pair_pretrain_path']))
                checkpoint = torch.load(self.hps['surf_net']['pair_pretrain_path'])
                self.pair.load_state_dict(checkpoint['state_dict'])
                print("=> loaded pair network pretrain checkpoint (epoch {})"
                    .format(checkpoint['epoch']))
            else:
                raise Exception("Pair network can not be restored.")
        
    def forward(self, x, tr_flag=False):
        logits = self.unary(x, logSoftmax=False).squeeze(1).permute(0, 2, 1)  
        logits = normalize_prob(logits)
        if self.pair is None:
            d_p = torch.zeros((x.size(0), x.size(-1)-1), dtype=torch.float32, requires_grad=False).cuda()
        else:
            self.pair.eval()
            d_p = self.pair(x)
     
        mean, sigma = gaus_fit(logits, tr_flag=tr_flag)
        output = newton_sol_pd(mean, sigma, self.w_comp, d_p)

        return output

class SurfSegNSBNet(torch.nn.Module):
    """
    ONly GPU version has been implemented!!!
    """
    def __init__(self, unary_model, hps):
        super(SurfSegNSBNet, self).__init__()
        self.unary = unary_model
        self.hps = hps
    def load_wt(self):
        if os.path.isfile(self.hps['surf_net']['resume_path']):
            print('loading surfNSBnet checkpoint: {}'.format(self.hps['surf_net']['resume_path']))
            checkpoint = torch.load(self.hps['surf_net']['resume_path'])
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded surfnet checkpoint (epoch {})"
                .format(checkpoint['epoch']))
        else:
            if os.path.isfile(self.hps['surf_net']['unary_pretrain_path']):
                print('loading unary network pretrain checkpoint: {}'.format(self.hps['surf_net']['unary_pretrain_path']))
                checkpoint = torch.load(self.hps['surf_net']['unary_pretrain_path'])
                self.unary.load_state_dict(checkpoint['state_dict'])
                print("=> loaded unary network pretrain checkpoint (epoch {})"
                    .format(checkpoint['epoch']))
            else:
                raise Exception("surf nsb network is not pretrained.")
        
    def forward(self, x):
        logits = self.unary(x, logSoftmax=False).squeeze(1).permute(0, 2, 1)  
        logits = normalize_prob(logits)
     
        mean, _ = gaus_fit(logits, tr_flag=self.training)
        return mean

if __name__ == "__main__":
    unary_model = FCN(num_classes=1, in_channels=1, depth=5, start_filts=1, up_mode="bilinear")
    pair_model = PairNet(num_classes=1, in_channels=1, depth=5, start_filts=1, up_mode="bilinear")
    surfnet = SurfSegNet(unary_model=unary_model, hps=None).cuda()
    x = torch.FloatTensor(np.random.random((2,1,512,400))).cuda()
    y = surfnet(x)
    print(y.size())

    module = UNet(num_classes=1, in_channels=1, depth=5, start_filts=1, up_mode="bilinear")
    x = torch.FloatTensor(np.random.random((2,1,512,400)))
    y = module(x)
    print(module)
    print(y.size())
    module = PairNet(num_classes=1, in_channels=1, depth=5, start_filts=1, up_mode="bilinear")
    y = module(x)
    print(module)
    print(y.size())