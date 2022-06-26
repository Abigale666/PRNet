import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.PRNet_Res2Net import PRNet
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F

# full_net_path='./full_net_path_21'
# resume=False
#
# def construct_print(out_str: str, total_length: int = 80):
#     if len(out_str) >= total_length:
#         extended_str = '=='
#     else:
#         extended_str = '=' * ((total_length - len(out_str)) // 2 - 4)
#     out_str = f" {extended_str}>> {out_str} <<{extended_str} "
#     print(out_str)

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record2, loss_record3, loss_record4, loss_record51, loss_record52 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            lateral_map_51, lateral_map_52, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
            # ---- loss function ----
            loss51 = structure_loss(lateral_map_51, gts)
            loss52 = structure_loss(lateral_map_52, gts)
            loss4 = structure_loss(lateral_map_4, gts)
            loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts)
            loss = loss2 + loss3 + loss4 + loss52 + loss51    # TODO: try different weights for loss
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record52.update(loss52.data, opt.batchsize)
                loss_record51.update(loss51.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-51: {:0.4f}, lateral-52: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record51.show(), loss_record52.show()))
    
    # state_dict = {
    #         'epoch': epoch,
    #         'net_state': model.state_dict(),
    #         'opti_state': optimizer.state_dict(),
    #     }
    # torch.save(state_dict, full_net_path)             # 1.保存整个模型: torch.save(net,'net.pkl')  2.只保存模型参数 torch.save(net.state_dict(),'net_parms.pkl')

    save_path = './snapshots/{}/'.format(opt.train_save)                      #修改训练好的模型的保存路径
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 10 == 0 or (epoch+1) > 20:
        torch.save(model.state_dict(), save_path + 'PRNet-%d.pth' % (epoch+1))           #修改模型名称
        print('[Saving Snapshot:]', save_path + 'PRNet-%d.pth'% (epoch+1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=40, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,                                     # 修改训练集路径
                        default='./data/TrainDataset/', help='path to train dataset')
    parser.add_argument('--train_save', type=str,                                     # 修改模型保存路径
                        default='PRNet')
    opt = parser.parse_args()

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = PRNet().cuda()

    # ---- flops and params ----
    # from utils.utils import CalParams
    # x = torch.randn(1, 3, 352, 352).cuda()
    # CalParams(lib, x)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)
    
    # if os.path.exists(full_net_path) and os.path.isfile(full_net_path):
    #     construct_print(f"Loading checkpoint '{full_net_path}'")
    #     checkpoint = torch.load(full_net_path)
    #
    # if resume:
    #     start_epoch = checkpoint['epoch']
    #     model.load_state_dict(checkpoint['net_state'])
    #     optimizer.load_state_dict(checkpoint['opti_state'])
    # else:
    #     start_epoch = 1

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    for epoch in range(0, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)
