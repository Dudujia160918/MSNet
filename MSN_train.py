import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import numpy as np
import glob
from data_loader import RandomCrop
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model import MSN,SSIM
import time

# ------- 1. define loss function --------

ssim_loss = SSIM(window_size=11,size_average=True)
bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5,d6, cx0, labels_v):

    loss0 = bce_loss(d0,labels_v) + ssim_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    loss4 = bce_loss(d4,labels_v)
    loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)
    loss00 = bce_loss(cx0,labels_v) + ssim_loss(cx0,labels_v)
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss00

    return loss0, loss


def main():
    model_name = "MSN"
    batch_size_unet = 5
    epoch_num = 301
    save_frq = 10
    # ------- 2. set the directory of training dataset --------

    data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
    tra_image_dir_fusion = "img1\\"
    tra_image_dir_mul = "img2\\"
    tra_label_dir = "GT2\\"
    image_ext = '.png'
    label_ext = '.png'
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

    tra_img_name_list_fusion = glob.glob(data_dir + tra_image_dir_fusion + '*' + image_ext)
    tra_img_name_list_mul = glob.glob(data_dir + tra_image_dir_mul + '*' + image_ext)
    tra_lbl_name_list = glob.glob(data_dir + tra_label_dir + '*' + label_ext)

    train_num = len(tra_img_name_list_fusion)
    print("---\ntrain images: ", train_num)
    print("train labels: ", len(tra_lbl_name_list), "\n---")

    #注意，这里的随机裁剪100，是保持fused图像尺寸不变（400），仅对100扩大到128的多光谱影像（关于扩大到128，只是我的一个随机，并没有做过多测试），进行随机裁剪为100，确保fusee图像范围更大
    salobj_dataset = SalObjDataset(tra_img_name_list_fusion, tra_img_name_list_mul, tra_lbl_name_list,transform=transforms.Compose([RandomCrop(100), ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size= batch_size_unet, drop_last=True,shuffle=True, num_workers=3)

    # ------- 3. define model --------
    # define the net
    net = MSN(3, 1).cuda()


    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.95)
    # ------- 5. training process --------
    print("---start training...")
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0

    txt = open("training loss.txt", "a+")
    txt.write("epoch" + "\t\t" + "losses_all" + "\t\t" + "losses_tar_all" + "\n")
    for epoch in range(0, epoch_num):
        net.train()
        start = time.time()
        for i, data in enumerate(salobj_dataloader):
            ite_num4val = ite_num4val + 1
            fusion, labels, Mul = data['fused'], data['label'], data['multi']
            fusion = fusion.type(torch.FloatTensor)
            Mul = Mul.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
            # wrap them in Variable
            fusion_v, labels_v, Mul_v = Variable(fusion.cuda(), requires_grad=False), \
                                         Variable(labels.cuda(),requires_grad=False), \
                                         Variable(Mul.cuda(), requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6, cx0= net(fusion_v, Mul_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6,cx0 , labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_unet, train_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
        scheduler.step()
        log = str(epoch+1) + "\t\t" + str(np.mean(running_loss / ite_num4val)) + "\t\t" + str(np.mean(running_tar_loss / ite_num4val)) + "\n"
        txt.write(log)
        txt.flush()

        if epoch>0 and epoch % 1 == 0:
            torch.save(net.state_dict(), model_dir + model_name+"_%d.pth" % (epoch))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0
        end = time.time()
        print("Time:", end - start)
if __name__ == "__main__":

    main()
