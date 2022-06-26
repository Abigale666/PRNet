import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
# from scipy import misc
import imageio
from lib.PRNet_Res2Net import PRNet
from utils.dataloader import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshots/PRNet/PRNet-40.pth')           #修改预训练模型的路径
#'CHAMELEON', 'CAMO','COD10K'
for _data_name in ['InsectCamouflage']:
    #data_path = '/content/drive/Shared drives/yuan2/dataset/Camouflage/TestDataset/{}/'.format(_data_name)
    data_path = '/data/home/scv0415/run/Abigale/data/TestDataset/{}/'.format(_data_name)
    save_path = './results/{}/'.format(_data_name)            #修改测试结果保存路径
    opt = parser.parse_args()
    model = PRNet()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res51, res52, res4, res3, res2 = model(image)
        res = res2
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # misc.imsave(save_path+name, res)
        imageio.imwrite(save_path+name, res)


# # 1.提取整个模型
# net2 = torch.load('net.pkl')
#
# # 2.只提取模型参数,要首先建立一个net3和net结构一致的网络
# net3 = Net(1,10,1)
# net3.load_state_dict(torch.load('net_parms.pkl'))
