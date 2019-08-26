from OCTDataset import OCTDataset
from torch.utils.data import DataLoader
import os

SURF = 1
tr_dir = "/home/leizhou/Documents/OCT/split_data_2D_400/train/"
img = "train_patch_aug.npy"
gt = "train_truth_aug.npy"
g_gt = "train_g_truth_surf_%s_50_aug.npy" % SURF

dataset = OCTDataset(surf=SURF, img_np=os.path.join(tr_dir, img), label_np=os.path.join(tr_dir, gt),
                     g_label_np=os.path.join(tr_dir, g_gt))
loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

for i in loader:
    print(i['img'].size(), i['gt'].size(), i['g_gt'].size())
    break