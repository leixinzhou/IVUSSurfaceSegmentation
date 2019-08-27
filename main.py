from torch.utils.data import DataLoader
from IVUSDataset import *
import network
import yaml
import argparse
import time
from tensorboardX import SummaryWriter
from torch import optim
from torch import nn
import shutil
import os

def roll_img(x, n):
    return torch.cat((x[:,:,:,-n:], x[:,:,:, :-n]), dim=-1)
def roll_pred(x, n):
    return torch.cat((x[:,-n:], x[:, :-n]), dim=-1)

def save_checkpoint(states,  path, filename='model_best.pth.tar'):
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint_name = os.path.join(path,  filename)
    torch.save(states, checkpoint_name)

# train


def train(model, criterion, optimizer, input_img_gt, hps):
    model.train()
    D = model(input_img_gt['img'])
    criterion_l1 = nn.L1Loss()
    # print(D.size(), input_img_gt['gt_g'].size())
    if hps['network'] == "UNet" or hps['network'] == "FCN":
        loss = criterion(D, input_img_gt['gt_g'].squeeze(-1))
    elif hps['network']=="PairNet":
        loss =  criterion(D, input_img_gt['gt_d'])
        
    elif hps['network']=="SurfNet" or hps['network']=="SurfSegNSBNet":
        loss =  criterion(D, input_img_gt['gt'])
        loss_l1 = criterion_l1(D, input_img_gt['gt'])
    else:
        raise AttributeError('Network not implemented!')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if hps['network']=="PairNet" or hps['network']=="SurfSegNSBNet":
        return loss_l1.detach().cpu().numpy()
    return loss.detach().cpu().numpy()
# val


def val(model, criterion, input_img_gt, hps):
    model.eval()
    D = model(input_img_gt['img'])
    criterion_l1 = nn.L1Loss()
    # print(output.size(), input_img_gt['gaus_gt'].size())
    if hps['network'] == "UNet" or hps['network'] == "FCN":
        loss = criterion(D, input_img_gt['gt_g'].squeeze(-1))
    elif hps['network']=="PairNet":
        loss =  criterion(D, input_img_gt['gt_d'])
        loss_l1 = criterion_l1(D, input_img_gt['gt_d'])
    elif hps['network']=="SurfNet" or hps['network']=="SurfSegNSBNet":
        loss =  criterion(D, input_img_gt['gt'])
        loss_l1 = criterion_l1(D, input_img_gt['gt'])
    else:
        raise AttributeError('Network not implemented!')
    if hps['network']=="PairNet" or hps['network']=="SurfSegNSBNet":
        return loss_l1.detach().cpu().numpy()
    return  loss.detach().cpu().numpy()
# learn




def learn(model, hps):
    since = time.time()
    writer = SummaryWriter(hps['learning']['checkpoint_path'])
    if torch.cuda.device_count() >= 1:
        # os.environ["CUDA_VISIBLE_DEVICES"] = hps['gpu'])
        model.cuda()
        # model = nn.DataParallel(model)
    else:
        raise NotImplementedError("CPU version is not implemented!")

    IVUSdiv = IVUSDivide(
        hps['learning']['data']['gt_dir_prefix'], hps['learning']['data']['img_dir_prefix'], tr_ratio=hps['learning']['data']['tr_ratio'], \
            val_ratio=hps['learning']['data']['val_ratio'])
    case_list = IVUSdiv(surf=hps['surf'], seed=hps['learning']['data']['seed'])
    print("tr nb: ", len(case_list['tr_list']))
    print(case_list['tr_list'])
    print("val nb: ", len(case_list['val_list']))
    print(case_list['val_list'])
    aug_dict = {"saltpepper": SaltPepperNoise(sp_ratio=0.05), 
                "Gaussian": AddNoiseGaussian(loc=0, scale=0.1),
                "cropresize": RandomCropResize(crop_ratio=0.9), 
                "circulateud": CirculateUD(),
                "mirrorlr":MirrorLR(), 
                "circulatelr": CirculateLR()}
    rand_aug = RandomApplyTrans(trans_seq=[aug_dict[i] for i in hps['learning']['augmentation']],
                                trans_seq_post=[NormalizeSTD()],
                                trans_seq_pre=[NormalizeSTD()])
    val_aug = RandomApplyTrans(trans_seq=[],
                                trans_seq_post=[NormalizeSTD()],
                                trans_seq_pre=[NormalizeSTD()])
    tr_dataset = IVUSDataset(case_list['tr_list'], transform=rand_aug)
    tr_loader = DataLoader(tr_dataset, shuffle=True,
                           batch_size=hps['learning']['batch_size'], num_workers=0)
    val_dataset = IVUSDataset(case_list['val_list'], transform=val_aug)
    val_loader = DataLoader(val_dataset, shuffle=False,
                            batch_size=hps['learning']['batch_size'], num_workers=0)

    optimizer = getattr(optim, hps['learning']['optimizer'])(
        [{'params': model.parameters(), 'lr': hps['learning']['lr']}])
    # scheduler = getattr(optim.lr_scheduler,
    #                     hps.learning.scheduler)(optimizer, factor=hps.learning.scheduler_params.factor,
    #                                             patience=hps.learning.scheduler_params.patience,
    #                                             threshold=hps.learning.scheduler_params.threshold,
    #                                             threshold_mode=hps.learning.scheduler_params.threshold_mode)
    try:
        loss_func = getattr(nn, hps['learning']['loss'])()
    except AttributeError:
        raise AttributeError(hps['learning']['loss']+" is not implemented!")
    # criterion_KLD = torch.nn.KLDivLoss()

    if os.path.isfile(hps['learning']['resume_path']):
        print('loading checkpoint: {}'.format(hps['learning']['resume_path']))
        checkpoint = torch.load(hps['learning']['resume_path'])
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(hps['learning']['resume_path']))

    epoch_start = 0
    best_loss = hps['learning']['best_loss']

    for epoch in range(epoch_start, hps['learning']['total_iterations']):
        # tr_loss_g = 0
        tr_loss_d = 0
        tr_mb = 0
        print("Epoch: " + str(epoch))
        for step, batch in enumerate(tr_loader):
            batch = {key: value.cuda() for (key, value) in batch.items() if "dir" not in key}
            m_batch_loss = train(model, loss_func, optimizer, batch, hps)
            # tr_loss_g += m_batch_loss[0]
            tr_loss_d += m_batch_loss
            tr_mb += 1
            print("         mini batch train loss: "+ "%.5e" % m_batch_loss)
        # epoch_tr_loss_g = tr_loss_g / tr_mb
        epoch_tr_loss_d = tr_loss_d / tr_mb
        # writer.add_scalar('data/train_loss_g', epoch_tr_loss_g, epoch)
        writer.add_scalar('data/train_loss_d', epoch_tr_loss_d, epoch)
        
        # print("     tr_loss_g: " + "%.5e" % epoch_tr_loss_g)
        print("     tr_loss_d: " + "%.5e" % epoch_tr_loss_d)
        # scheduler.step(epoch_tr_loss)

        # val_loss_g = 0
        val_loss_d = 0
        val_mb = 0
        for step, batch in enumerate(val_loader):
            batch = {key: value.cuda() for (key, value) in batch.items() if "dir" not in key}
            m_batch_loss = val(model, loss_func, batch, hps)
            # val_loss_g += m_batch_loss[0]
            val_loss_d += m_batch_loss
            val_mb += 1
            print("         mini batch val loss: "+ "%.5e" % m_batch_loss)
        # epoch_val_loss_g = val_loss_g / val_mb
        epoch_val_loss_d = val_loss_d / val_mb
        # writer.add_scalar('data/val_loss_g', epoch_val_loss_g, epoch)
        writer.add_scalar('data/val_loss_d', epoch_val_loss_d, epoch)
        # print("     val_loss_g: " + "%.5e" % epoch_val_loss_g)
        print("     val_loss_d: " + "%.5e" % epoch_val_loss_d)

        if epoch_val_loss_d < best_loss:
            best_loss = epoch_val_loss_d
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict()
                },
                path=hps['learning']['checkpoint_path'],
            )

    writer.export_scalars_to_json(os.path.join(
        hps['learning']['checkpoint_path'], "all_scalars.json"))
    writer.close()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

def infer(model, hps):
    since = time.time()
    if torch.cuda.device_count() >= 1:
        # os.environ["CUDA_VISIBLE_DEVICES"] = hps['gpu'])
        model.cuda()
        # model = nn.DataParallel(model)
    else:
        raise NotImplementedError("CPU version is not implemented!")
    IVUStest = IVUSTest(
        hps['test']['data']['gt_dir_prefix'], hps['test']['data']['img_dir_prefix'])
    case_list = IVUStest(surf=hps['surf'])
    rand_aug = RandomApplyTrans(trans_seq=[],
                                trans_seq_post=[NormalizeSTD()],
                                trans_seq_pre=[NormalizeSTD()])
    test_dataset = IVUSDataset(case_list, transform=rand_aug, gaus_gt=False)
    test_loader = DataLoader(test_dataset, shuffle=False,
                           batch_size=hps['test']['batch_size'], num_workers=0)
    
    if os.path.isfile(hps['test']['resume_path']):
        print('loading checkpoint: {}'.format(hps['test']['resume_path']))
        checkpoint = torch.load(hps['test']['resume_path'])
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(hps['test']['resume_path']))
    model.eval()
    if hps['network']=="PairNet":
        pred_l1 = []
        pred_dummy = []
        for step, batch in enumerate(test_loader):
            pred = np.zeros(255, dtype=np.float32)
            batch_gt_d = batch['gt_d'].squeeze().detach().cpu().numpy()
            batch_gt = batch['gt'].squeeze().detach().cpu().numpy()
            # print(batch_gt_d)
            # print(batch_gt)
            # break
            batch_img = batch['img'].float().cuda()
            pred_tmp = model(batch_img)
            pred = pred_tmp.squeeze().detach().cpu().numpy()
            # convert to cart
            img = plt.imread(batch['img_dir'][0])
            phy_radius = 0.5*np.sqrt(np.average(np.array(img.shape)**2)) - 1
            cartpolar = CartPolar(np.array(img.shape)/2.,
                                phy_radius, 256, 128)
            img_polar = cartpolar.img2polar(img)
            fig, axes = plt.subplots(2,1)
            axes[0].imshow(img_polar, cmap="gray", aspect='auto')
            axes[0].plot(batch_gt, 'r', label='gt')
            axes[0].legend()
            axes[1].plot(pred, 'r', label='diff pred')
            axes[1].plot(batch_gt_d, 'b', label='diff gt')
            axes[1].legend()
            # pred = cartpolar.gt2cart(pred)
            if not os.path.isdir(hps['test']['pred_dir']):
                os.mkdir(hps['test']['pred_dir'])
            pred_dir = os.path.join(hps['test']['pred_dir'],batch['gt_dir'][0].split("/")[-1].replace(".txt", ".png"))
            fig.savefig(pred_dir)
            plt.close()
            pred_l1.append(np.mean(np.abs(batch_gt_d-pred)))
            pred_dummy.append(np.mean(np.abs(batch_gt_d)))
            # pred = np.transpose(np.stack(pred, axis=0))
        
            # np.savetxt(pred_dir, pred, delimiter=',')
        print("Test done!")
        pred_l1_mean = np.mean(np.array(pred_l1))
        pred_l1_std = np.std(np.array(pred_l1))
        dummy_l1_mean = np.mean(np.array(pred_dummy))
        dummy_l1_std = np.std(np.array(pred_dummy))
        print("test L1: ", "%.5e" % pred_l1_mean)
        print("test L1 std: ", "%.5e" % pred_l1_std)
        print("test dummy L1: ", "%.5e" % dummy_l1_mean)
        print("test dummy L1 std: ", "%.5e" % dummy_l1_std)
        np.savetxt(os.path.join(hps['test']['pred_dir'], "results.txt"), [pred_l1_mean, pred_l1_std, dummy_l1_mean, dummy_l1_std])
    elif hps['network']=="SurfSegNSBNet":
        rept_nb = 8
        for step, batch in enumerate(test_loader):
            pred = np.zeros(256, dtype=np.float32)
            for shift_nb in range(rept_nb):
                batch_img = batch['img'].float().cuda()
                if shift_nb==0:
                    pred_tmp = model(batch_img)
                    pred_tmp = pred_tmp.squeeze().detach().cpu().numpy()
                    pred += pred_tmp
                else:
                    shift = int(shift_nb*256/rept_nb)
                    batch_img = roll_img(batch_img, shift)
                    pred_tmp = model(batch_img)
                    pred_tmp = roll_pred(pred_tmp, -shift)
                    pred_tmp = pred_tmp.squeeze().detach().cpu().numpy()
                    pred += pred_tmp
            pred = 1.*pred/rept_nb
            # convert to cart
            img = plt.imread(batch['img_dir'][0])
            phy_radius = 0.5*np.sqrt(np.average(np.array(img.shape)**2)) - 1
            cartpolar = CartPolar(np.array(img.shape)/2.,
                                phy_radius, 256, 128)
            pred = cartpolar.gt2cart(pred)
            if not os.path.isdir(hps['test']['pred_dir']):
                os.mkdir(hps['test']['pred_dir'])
            pred_dir = os.path.join(hps['test']['pred_dir'],batch['gt_dir'][0].split("/")[-1])
            pred = np.transpose(np.stack(pred, axis=0))
        
            np.savetxt(pred_dir, pred, delimiter=',')
        print("Test done!")


def main():
    # read configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hyperparams', default='./para/hparas_unet.json',
                        type=str, metavar='FILE.PATH',
                        help='path to hyperparameters setting file (default: ./para/hparas_unet.json)')

    args = parser.parse_args()
    try:
        with open(args.hyperparams, "r") as config_file:
            hps = yaml.load(config_file)
    except IOError:
        print('Couldn\'t read hyperparameter setting file')
    if hps['network'] == "UNet" or hps['network'] == "FCN":
        model = getattr(network, hps['network'])(num_classes=1, in_channels=1, depth=hps['unary_network']['depth'],
                 start_filts=hps['unary_network']['start_filters'], up_mode=hps['unary_network']['up_mode'])
        if os.path.isfile(hps['unary_network']['resume_path']):
            print('loading unary network checkpoint: {}'.format(hps['unary_network']['resume_path']))
            checkpoint = torch.load(hps['unary_network']['resume_path'])
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded unary checkpoint (epoch {})"
                .format(checkpoint['epoch']))
        else:
            print("=> no unary network checkpoint found at '{}'".format(hps['unary_network']['resume_path']))

    elif hps['network']=="PairNet":
        model = getattr(network, hps['network'])(num_classes=1, in_channels=1, depth=hps['pair_network']['depth'],
                            start_filts=hps['pair_network']['start_filters'], up_mode=hps['pair_network']['up_mode'], 
                            col_len=hps['pair_network']['col_len'], fc_inter=hps['pair_network']['fc_inter'], 
                            left_nbs=hps['pair_network']['left_nbs'])
        print(model)
        # sys.exit(0)
        if os.path.isfile(hps['pair_network']['resume_path']):
            print('loading pair network checkpoint: {}'.format(hps['pair_network']['resume_path']))
            checkpoint = torch.load(hps['pair_network']['resume_path'])
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})"
                .format(checkpoint['epoch']))
        else:
            print("=> no pair network checkpoint found at '{}'".format(hps['pair_network']['resume_path']))

    elif hps['network']=="SurfSegNSBNet":
        model_u = getattr(network, hps['surf_net']['unary_network'])(num_classes=1, in_channels=1, depth=hps['unary_network']['depth'],
                 start_filts=hps['unary_network']['start_filters'], up_mode=hps['unary_network']['up_mode'])
        model = getattr(network, hps['network'])(unary_model=model_u, hps=hps)
        model.load_wt()
    else:
        raise AttributeError('Network not implemented!')

    
    if hps['test']['mode']:
        infer(model, hps)
    else:
        try:
            learn(model, hps)
        except KeyboardInterrupt:
            torch.save(model.state_dict(), os.path.join(
                hps['learning']['checkpoint_path'], 'INTERRUPTED.pth'))
            print('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)



if __name__ == '__main__':
    main()