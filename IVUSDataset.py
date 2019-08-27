import sys
sys.path.append('../')
from CartToPolar import *
from AugSurfSeg import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import torch
import os, random
import matplotlib.pyplot as plt




ROW_LEN = 128
COL_LEN = 256
SIGMA = 15.


def gaus_pdf(x_arry, mean, sigma, A=1.):
    pdf_array = np.empty_like(x_arry, dtype=np.float16)
    for i in range(x_arry.shape[0]):
        pdf_array[i] = A * np.exp((-(x_arry[i] - mean)**2)/(2*sigma**2))
    pdf_array = np.exp(pdf_array) / np.sum(np.exp(pdf_array), axis=0)
    return pdf_array


# define a look-up table
# lookup table
LU_TABLE = np.empty((ROW_LEN, ROW_LEN), dtype=np.float16)
x_range = np.arange(ROW_LEN).astype(np.float16)
for i in range(ROW_LEN):
    prob = gaus_pdf(x_range, i, SIGMA)
    LU_TABLE[i, ] = prob


class IVUSDivide(object):
    def __init__(self, gt_dir_prefix, img_dir_prefix,  tr_ratio=0.9, val_ratio=0.1):
        self.gt_dir_prefix = gt_dir_prefix
        self.img_dir_prefix = img_dir_prefix
        self.tr_ratio = tr_ratio
        self.val_ratio = val_ratio

    def __call__(self, surf=['lum'], seed=0):
        """
        surf: 'lum' or 'med'
        """
        surf = surf[0]
        gt_list = os.listdir(self.gt_dir_prefix)
        list_gt = list(set([i[4:-4] for i in gt_list]))
        list_gt.sort()
        random.seed(seed)
        random.shuffle(list_gt)
        tr_case_nb = int(len(list_gt)*self.tr_ratio)
        val_case_nb = int(len(list_gt)*self.val_ratio)
        tr_list = [{'img_dir': os.path.join(self.img_dir_prefix, '%s.png' % i),
                    'gt_dir': os.path.join(self.gt_dir_prefix, '%s_%s.txt' % (surf, i))} for i in list_gt[:tr_case_nb]]
        val_list = [{'img_dir': os.path.join(self.img_dir_prefix, '%s.png' % i),
                     'gt_dir': os.path.join(self.gt_dir_prefix, '%s_%s.txt' % (surf, i))} for i in list_gt[-val_case_nb:]]
        return {'tr_list': tr_list, 'val_list': val_list}

class IVUSTest(object):
    def __init__(self, gt_dir_prefix, img_dir_prefix):
        self.gt_dir_prefix = gt_dir_prefix
        self.img_dir_prefix = img_dir_prefix
    def __call__(self, surf=['lum']):
        surf = surf[0]
        gt_list = os.listdir(self.gt_dir_prefix)
        list_gt = list(set([i[4:-4] for i in gt_list]))
        list_gt.sort()
        test_list = [{'img_dir': os.path.join(self.img_dir_prefix, '%s.png' % i),
                    'gt_dir': os.path.join(self.gt_dir_prefix, '%s_%s.txt' % (surf, i))} for i in list_gt]
        return test_list

class IVUSDataset(Dataset):
    """convert 3d dataset to Dataset."""

    def __init__(self, case_list, gaus_gt=True, transform=None, batch_nb=None):
        """
        Args:
            case_list (of dictionary): all  cases
            gaus_gt: output containes gaus_gt or not (default True)
            transform (callable, optional): Optional transform to be applied on a sample.
            batch_nb (int, optional): can use to debug.
        """
        self.case_list = case_list
        self.bn = batch_nb
        self.gaus_gt = gaus_gt
        self.transform = transform

    def __len__(self):
        if self.bn is None:
            return len(self.case_list)
        else:
            return self.bn

    def __getitem__(self, idx):
        img_dir = self.case_list[idx]['img_dir']
        gt_dir = self.case_list[idx]['gt_dir']
        img = plt.imread(img_dir)
        gt = np.loadtxt(gt_dir, delimiter=',')
        # convert to polar
        phy_radius = 0.5*np.sqrt(np.average(np.array(img.shape)**2)) - 1
        cartpolar = CartPolar(np.array(img.shape)/2.,
                              phy_radius, COL_LEN, ROW_LEN)
        polar_img = cartpolar.img2polar(img)
        polar_gt = cartpolar.gt2polar(gt)
        # polar_img = polar_img.transpose()
        polar_gt = np.expand_dims(polar_gt, axis=1)
        input_img_gt = {'img': polar_img, 'gt': polar_gt}
        # apply augmentation transform
        if self.transform is not None:
            input_img_gt = self.transform(input_img_gt)

        # print(input_img_gt['gt'].shape)
        if self.gaus_gt:
            polar_gt_gaus = np.empty_like(polar_img)
            # print(polar_gt_gaus.shape, LU_TABLE.shape, input_img_gt['gt'].shape)
            for i in range(COL_LEN):
                polar_gt_gaus[:, i] = LU_TABLE[int(
                    np.clip(np.around(input_img_gt['gt'][i, 0]), 0, ROW_LEN-1)), ]
            input_img_gt['gt_g'] = polar_gt_gaus
        input_img_gt['img'] = np.expand_dims(input_img_gt['img'], axis=0)
        input_img_gt = {key:value.astype(np.float32) for (key, value) in input_img_gt.items()}
        # add gt_dir to the output
        input_img_gt['gt_dir'] = gt_dir
        input_img_gt['img_dir'] = img_dir
        # add gt_d to the output
        input_img_gt['gt'] = np.squeeze(input_img_gt['gt'], axis=1)
        input_img_gt['gt_d'] = input_img_gt['gt'][:-1] - input_img_gt['gt'][1:]
    
        return input_img_gt

if __name__ == "__main__":
    gt_dir_prefix = "../../IVUS/Training_Set/Data_set_B/LABELS/"
    img_dir_prefix = "../../IVUS/Training_Set/Data_set_B/DCM/"
    tr_ratio = 0.1
    val_ratio = 0.1
    seed = 0 
    IVUSdiv = IVUSDivide(
        gt_dir_prefix, img_dir_prefix, tr_ratio=tr_ratio, \
            val_ratio=val_ratio)
    case_list = IVUSdiv(surf=['lum'], seed=seed)
    aug_dict = {"saltpepper": SaltPepperNoise(sp_ratio=0.05), 
                "Gaussian": AddNoiseGaussian(loc=0, scale=0.1),
                "cropresize": RandomCropResize(crop_ratio=0.9), 
                "circulateud": CirculateUD(),
                "mirrorlr":MirrorLR(), 
                "circulatelr": CirculateLR()}
    rand_aug = RandomApplyTrans(trans_seq=[aug_dict[i] for i in aug_dict],
                                trans_seq_post=[NormalizeSTD()],
                                trans_seq_pre=[NormalizeSTD()])
    tr_dataset = IVUSDataset(case_list['tr_list'], transform=rand_aug)
    tr_loader = DataLoader(tr_dataset, shuffle=False,
                           batch_size=1, num_workers=0)
    for step, batch in enumerate(tr_loader):
        img = batch['img'].squeeze().detach().numpy()
        gt = batch['gt'].squeeze().detach().numpy()
        gt_d = batch['gt_d'].squeeze().detach().numpy()
        gt_g = batch['gt_g'].squeeze().detach().numpy()
        break
    _, axes = plt.subplots(1,3)
    axes[0].imshow(img, cmap='gray')
    axes[0].plot(gt)
    axes[1].plot(gt_d)
    axes[2].imshow(gt_g)
    plt.show()

