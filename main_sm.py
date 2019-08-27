from torch.utils.data import DataLoader
from IVUSDataset import *
import network 
import argparse
import time
from tensorboardX import SummaryWriter
from torch import optim
from torch import nn
import shutil
import os
import yaml
from collections import OrderedDict


def roll_img(x, n):
    return torch.cat((x[:,:,:,-n:], x[:,:,:, :-n]), dim=-1)
def roll_pred(x, n):
    return torch.cat((x[:,-n:], x[:, :-n]), dim=-1)

def save_checkpoint(states,  path, filename='model_best.pth.tar'):
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint_name = os.path.join(path,  filename)
    torch.save(states, checkpoint_name)
# def load_my_state_dict(new_state_dict, old_state_dict):
#     for name, param in old_state_dict.items():
#         if name not in new_state_dict:
#                 continue
#         if isinstance(param, nn.Parameter):
#             # backwards compatibility for serialized parameters
#             param = param.data
#         new_state_dict[name].copy_(param)
def remove_module_prefix_by_para(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict
# train


def train(model, criterion, optimizer, input_img_gt, hps):
    model.train()
    output = model(input_img_gt['img'])
    criterion_l1 = nn.L1Loss()
    
    if hps['learning']['loss'] == "MSELoss" or hps['learning']['loss'] == "L1Loss":
        loss = criterion(output, input_img_gt['gt'])
        loss_l1 = criterion_l1(output, input_img_gt['gt'])
    else:
        raise AttributeError('Loss not implemented!')
    # watch l1
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss_l1.detach().cpu().numpy()
# val


def val(model, criterion, input_img_gt, hps):
    model.eval()
    output = model(input_img_gt['img'])
    criterion_l1 = nn.L1Loss()

    if hps['learning']['loss'] == "MSELoss" or hps['learning']['loss'] == "L1Loss":
        loss = criterion(output, input_img_gt['gt'])
        loss_l1 = criterion_l1(output, input_img_gt['gt'])
    else:
        raise AttributeError('Loss not implemented!')
    
    return loss_l1.detach().cpu().numpy()
# learn


def learn(model, hps):
    since = time.time()
    writer = SummaryWriter(hps['learning']['checkpoint_path'])
    print(torch.cuda.device_count())
    if torch.cuda.device_count() >= 1:
        # os.environ["CUDA_VISIBLE_DEVICES"] = hps['gpu'])
        model.cuda()
        # model = nn.DataParallel(model)

    else:
        raise NotImplementedError("CPU version is not implemented!")

    IVUSdiv = IVUSDivide(
        hps['learning']['data']['gt_dir_prefix'], hps['learning']['data']['img_dir_prefix'], tr_ratio=hps['learning']['data']['tr_ratio'])
    case_list = IVUSdiv(surf=hps['surf'], seed=hps['learning']['data']['seed'])
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
    # The parameters are distributed in three parts: U_net, P_net, w_comp. Here we fix the P_net.
    optimizer_unary = getattr(optim, hps['learning']['optimizer'])(
        [{'params': model.unary.parameters(), 'lr': hps['learning']['lr_unary']}
         ])
    optimizer_pair = getattr(optim, hps['learning']['optimizer'])(
        [{'params': model.w_comp, 'lr': hps['learning']['lr_pair']}
         ])
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

    epoch = 0
    best_loss = hps['learning']['best_loss']

    for epoch_tmp in range(0, hps['learning']['total_iterations']):
        for epoch_smoother_tmp in range(0, hps['learning']['pair_iterations']):
            tr_loss = 0
            tr_mb = 0
            for step, batch in enumerate(val_loader):
                batch = {key: value.float().cuda() for (key, value) in batch.items() if "dir" not in key}
                m_batch_loss = train(model, loss_func, optimizer_pair, batch, hps)
                tr_loss += m_batch_loss
                tr_mb += 1
            epoch_tr_loss = tr_loss / tr_mb
            writer.add_scalar('data/train_loss', epoch_tr_loss, epoch)
            w_comp = model.w_comp.detach().cpu().numpy()
            writer.add_scalar('data/w_comp', w_comp)
            print("Epoch: " + str(epoch))
            print("     tr_loss pair: " + "%.5e" % epoch_tr_loss + " w_comp: " + "%.5e" % w_comp)
            epoch += 1

            if epoch_tr_loss < best_loss:
                best_loss = epoch_tr_loss
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best_loss': best_loss
                    },
                    path=hps['learning']['checkpoint_path'],
                )
        for epoch_unet_tmp in range(0, hps['learning']['unary_iterations']):
            tr_loss = 0
            tr_mb = 0
            for step, batch in enumerate(tr_loader):
                batch = {key: value.float().cuda() for (key, value) in batch.items() if "dir" not in key}
                m_batch_loss = train(model, loss_func, optimizer_unary, batch, hps)
                tr_loss += m_batch_loss
                tr_mb += 1
            epoch_tr_loss = tr_loss / tr_mb
            writer.add_scalar('data/train_loss', epoch_tr_loss, epoch)
            print("Epoch: " + str(epoch))
            print("     tr_loss: " + "%.5e" % epoch_tr_loss)
            epoch += 1
            val_loss = 0
            val_mb = 0
            for step, batch in enumerate(val_loader):
                batch = {key: value.float().cuda() for (key, value) in batch.items() if "dir" not in key}
                m_batch_loss = val(model, loss_func,  batch, hps)
                val_loss += m_batch_loss
                val_mb += 1
            epoch_val_loss = val_loss / val_mb
            writer.add_scalar('data/val_loss', epoch_val_loss, epoch)
            print("     val_loss: " + "%.5e" % epoch_val_loss)
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best_loss': best_loss
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
    print(model.w_comp.detach().cpu().numpy())
        # batch_img = batch['img'].float().cuda()
        # pred = model(batch_img, U_net_only=hps.network.unet_only)
        # pred = pred.squeeze().detach().cpu().numpy()
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
    time_elapsed = time.time() - since
    print('Inference complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    time_slice = 1.*time_elapsed/step
    print("Time for each slice: %e" % time_slice)



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
    

    if hps['network']=="SurfSegNet":
        model_u = getattr(network, hps['surf_net']['unary_network'])(num_classes=1, in_channels=1, depth=hps['unary_network']['depth'],
                 start_filts=hps['unary_network']['start_filters'], up_mode=hps['unary_network']['up_mode'])
        if os.path.isfile(hps['surf_net']["pair_pretrain_path"]):
            model_p = PairNet(num_classes=1, in_channels=1, depth=hps['pair_network']['depth'],
                            start_filts=hps['pair_network']['start_filters'], up_mode=hps['pair_network']['up_mode'], 
                            col_len=hps['pair_network']['col_len'], fc_inter=hps['pair_network']['fc_inter'], 
                            left_nbs=hps['pair_network']['left_nbs'])
        else:
            model_p = None
        model = getattr(network, hps['network'])(unary_model=model_u, hps=hps, pair_model=model_p, wt_init=hps['surf_net']['wt_init'])
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
