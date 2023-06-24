import os
from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from model import MSN
from PIL import Image
import glob
from data_loader import ToTensorLab
from data_loader import SalObjDataset

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def save_output(image_name,pred, d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    imo.save(d_dir+imidx+'.png')

def main():

    # --------- 1. get image path and name ---------
    model_name='MSN'#u2netp
    epoch = "280"

    image_dir_pan = os.path.join(os.getcwd(), 'test_data', 'test_images1')
    image_dir_mul = os.path.join(os.getcwd(), 'test_data', 'test_images2')

    prediction_dir1 = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name,
                             model_name + "_" + str(epoch) + '.pth')

    img_name_list_pan = glob.glob(image_dir_pan + os.sep + '*')
    img_name_list_mul = glob.glob(image_dir_mul + os.sep + '*')
    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list_pan, img_name_list_mul, [],
                                        transform=transforms.Compose([ToTensorLab(flag=0)]), test=True)
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='MSN'):

        net = MSN(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()


    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        inputs_test,inputs_test_lbp,labels = data_test['image'], data_test['LBP'],data_test['label']

        inputs_test = inputs_test.type(torch.FloatTensor)
        inputs_test = Variable(inputs_test).cuda()
        inputs_test_lbp = inputs_test_lbp.type(torch.FloatTensor)
        inputs_test_lbp = Variable(inputs_test_lbp).cuda()
        d0, d1, d2, d3, d4, d5, d6, cx0 = net(inputs_test, inputs_test_lbp)


        # normalization
        pred = d0[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir1):
            os.makedirs(prediction_dir1, exist_ok=True)
        save_output(img_name_list_pan[i_test],pred,prediction_dir1)

        del d0, d1,d2,d3,d4,d5,d6,cx0,inputs_test,inputs_test_lbp,labels

if __name__ == "__main__":
    main()
