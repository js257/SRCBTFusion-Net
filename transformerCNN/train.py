from torch.utils.data import DataLoader
import torch.nn as nn
import loss
from metircs import Evaluator
import numpy as np
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.cuda.amp import autocast
from read_data import *
from early_stopping import EarlyStopping
from VITCNN import VisionTransformer as ViT_seg
from VITCNN import CONFIGS as CONFIGS_ViT_seg
import time


def main():
    best_iou = 0
    epoch1 = 60
    min_lr = 0.0001
    weight_decay = 0.0005 # 权重衰减
    LR = 0.0001
    batch_size = 16
    vit_name = 'R50-ViT-B_16'
    num_class = 6
    img_size = 256
   

    train_time = []
    p_train_time = []

    val_time = []
    p_val_time = []
    train_voc_dir = 'E:/cjs/Semantic/dataset/train'
    val_voc_dir = 'E:/cjs/Semantic/dataset/val'

    out_file = './checkpoint/'

    try:
        os.makedirs(out_file)
        os.makedirs(out_file + '/')
    except OSError:
        pass

    crop_size = (img_size, img_size)
    voc_train = VOCSegDataset(True, crop_size, train_voc_dir)
    voc_test = VOCSegDataset(False, crop_size, val_voc_dir)

    data_train_loader = torch.utils.data.DataLoader(voc_train, batch_size, pin_memory=True, shuffle=True,drop_last=True)
    data_val_loader = torch.utils.data.DataLoader(voc_test, batch_size, pin_memory=True, drop_last=True)

    config_vit = CONFIGS_ViT_seg[vit_name]
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / batch_size), int(img_size / batch_size))
    model = ViT_seg(config_vit, img_size=img_size, num_classes=num_class)
 
    device = torch.device('cuda')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    pretrained_dict = torch.load('E:/cjs/Semantic/transformerCNN/checkpoint/256/postdam/cjsnet-block8.pth')
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()
    }

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    #
    for p in model.parameters():
        p.requires_grad = True

    # model.load_from(weights=np.load(config_vit.pretrained_path))

    optimizer1 = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    scheduler1 = lr_scheduler.CosineAnnealingLR(optimizer1,
                                                T_max=epoch1,
                                                eta_min=min_lr)
    criterion1 = loss.JointEdgeSegLoss()
    evaluator = Evaluator(6)
    scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=10, verbose=True)


    for i in range(epoch1):
        print('Epoch [%d/%d]' % (i+1,epoch1))
        model.train()
        start = time.time()
        for _, data in tqdm(enumerate(data_train_loader), total=len(data_train_loader), desc='train'):
            img, label = data
            img, label = img.cuda(), label.cuda()
            optimizer1.zero_grad()

            with autocast():
                seg_out= model(img)
                loss_all = criterion1(seg_out, label)

            main_loss = loss_all['seg_loss']
            main_loss += loss_all['dual_loss']
            scaler.scale(main_loss).backward()
            scaler.step(optimizer1)
            scaler.update()
        scheduler1.step()
        #################################
        end = time.time()
        train_time.append(end - start)
        print('训练时间:{:.2f}m'.format((end - start)/60))
        f = open('logs/result_edg_v_50.txt', 'a')
        f.write('每一个周期训练时间: %.2f\n' % ((end - start)/60))
        f.close()
        #################################
        if early_stopping.early_stop:
            break

        with torch.no_grad():
            model.eval()
            evaluator.reset()
            start = time.time()
            for j, val_data in tqdm(enumerate(data_val_loader), total=len(data_val_loader), desc='val'):
                img_val, label_val = val_data
                img_val, label_val = img_val.cuda(), label_val.cuda()
                # img_val = img_val.permute(0, 3, 1, 2)

                with torch.no_grad():
                   seg_predict= model(img_val)
                seg_predict = torch.softmax(seg_predict, dim=1)
                _, seg_predict = torch.max(seg_predict, dim=1)
                #################################
            end = time.time()
            val_time.append(end - start)
            print('测试时间:{:.2f}m'.format((end - start) / 60))
            f = open('logs/result_edg_v_50.txt', 'a')
            f.write('每一个周期测试时间: %.2f\n' % ((end - start) / 60))
            f.close()
            #################################

            acc = evaluator.Pixel_Accuracy_Class()
            mIou = evaluator.Mean_Intersection_over_Union()
            precision = evaluator.Precision()
            fscore = evaluator.F1score()
            macc = evaluator.Pixel_Accuracy()
            miou = np.nanmean(mIou)

            sum_moiu = mIou[0] + mIou[1] + mIou[2] + mIou[3] + mIou[4]
            if sum_moiu >= best_iou:
                best_iou = sum_moiu
                f = open('logs/result_edg_v_50.txt', 'a')
                f.write("======================%d======================\n" % i)
                f.write('Impervious surfaces iou %.4f\n' % mIou[0])
                f.write('Building iou %.4f\n' % mIou[1])
                f.write('Low vegetation iou %.4f\n' % mIou[2])
                f.write('Tree iou %.4f\n' % mIou[3])
                f.write('Car iou %.4f\n' % mIou[4])
                f.write('background iou %.4f\n' % mIou[5])

                f.write('Impervious surfaces acc %.4f\n' % acc[0])
                f.write('Building acc %.4f\n' % acc[1])
                f.write('Low vegetation acc %.4f\n' % acc[2])
                f.write('Tree acc %.4f\n' % acc[3])
                f.write('Car acc %.4f\n' % acc[4])
                f.write('background acc %.4f\n' % acc[5])

                f.write('Impervious surfaces pre %.4f\n' % precision[0])
                f.write('Building pre %.4f\n' % precision[1])
                f.write('Low vegetation pre %.4f\n' % precision[2])
                f.write('Tree pre %.4f\n' % precision[3])
                f.write('Car pre %.4f\n' % precision[4])
                f.write('backgroundpre %.4f\n' % precision[5])

                f.write('Impervious surfaces F1 %.4f\n' % fscore[0])
                f.write('Building F1 %.4f\n' % fscore[1])
                f.write('Low vegetation F1 %.4f\n' % fscore[2])
                f.write('Tree F1 %.4f\n' % fscore[3])
                f.write('Car F1 %.4f\n' % fscore[4])
                f.write('background F1 %.4f\n' % fscore[5])

                los_s = 1 - miou
                macc = evaluator.Pixel_Accuracy()
                mpre = np.nanmean(precision)
                mf1 = np.nanmean(fscore)

                sum_acc = acc[0] + acc[1] + acc[2] + acc[3] + acc[4]
                sum_F = fscore[0] + fscore[1] + fscore[2] + fscore[3] + fscore[4]
                f.write('miou: %.4f\n' % miou)
                f.write('macc: %.4f\n' % macc)
                f.write('前5个miou平均: %.4f\n' % float(sum_moiu / 5))
                f.write('前5个macc平均: %.4f\n' % float(sum_acc / 5))
                f.write('前5个F1平均: %.4f\n' % float(sum_F / 5))
                f.write('mpre: %.4f\n' % mpre)
                f.write('mf1: %.4f\n' % mf1)
                f.write('loss: %.4f\n' % los_s)

                f.close()

                print("======================%d======================" % i)
                print('地面iou %.4f' % mIou[0])
                print('建筑 %.4f' % mIou[1])
                print('低植被 %.4f' % mIou[2])
                print('树 %.4f' % mIou[3])
                print('车 %.4f' % mIou[4])
                print('背景 %.4f' % mIou[5])

                print('miou: %.4f' % miou)
                print('acc: %.4f' % macc)
                print("======================保存模型参数======================")
                print('前5个miou平均 %.4f' % float(sum_moiu / 5))
                print('前5个macc平均 %.4f' % float(sum_acc / 5))
                print('前5个F1平均 %.4f' % float(sum_F / 5))
                print('loss: %.4f' % los_s)

                torch.save(model.state_dict(), '%s/' % out_file + 'netG.pth')
                ##################### 模型保存 ##########################
        early_stopping(1 - miou, model, '%s/' % out_file + 'netG.pth')
        if early_stopping.early_stop:
            break

    ############################### 混淆矩阵 #######################################
    ##############################################################################
    p_train_time.append(train_time)
    p_val_time.append(val_time)
    f = open('logs/result_edg_v_50.txt', 'a')
    f.write('平均训练时间: %.4f\n' % np.mean(p_train_time))
    f.write('平均测试时间: %.4f\n' % np.mean(p_val_time))
    f.close()

 
    


if __name__ == '__main__':
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    batch_size = 16
    vit_name = 'R50-ViT-B_16'
    num_class = 6
    img_size = 256

    # voc_test = RemoteSensingDataset(False, img_transforms)
    # if not args.deterministic:
    #     cudnn.benchmark = True
    #     cudnn.deterministic = False
    # else:
    #     cudnn.benchmark = False
    #     cudnn.deterministic = True
    #
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    # model = MA_Net(3, num_class).cuda()
    config_vit = CONFIGS_ViT_seg[vit_name]
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / batch_size), int(img_size / batch_size))
    model = ViT_seg(config_vit, img_size=img_size, num_classes=num_class).cuda()
    x = torch.randn(16, 3, 256, 256).cuda()
    x = model(x)
    # main()







