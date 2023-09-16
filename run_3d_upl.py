from utils import parse_config, set_random
from unet3d import UNet3d
from torch.utils.data import DataLoader
import torch
import SimpleITK as sitk
import os
from metrics import dice_eval,assd_eval
import numpy as np
import argparse
import monai
from monai.transforms import (Compose,LoadImaged,EnsureChannelFirstd,ScaleIntensityd,
                              RandSpatialCropd,ToTensord,AsDiscreted,SpatialPadd)
from monai.inferers import sliding_window_inference
def get_datalist(path_img,path_lab,data_transform):
    img_path = []
    lab_path = []
    imgs = os.listdir(path_img)
    labs = os.listdir(path_lab)
    imgs.sort()
    labs.sort()
    for img in imgs:
        img_dir = os.path.join(path_img,img)
        img_path.append(img_dir)
    for lab in labs:
        lab_dir = os.path.join(path_lab,lab)
        lab_path.append(lab_dir)

    assert len(img_path) == len(lab_path)
    data_dict = [{'image':image,'label':label,'name':name} for image,label,name in zip(img_path,lab_path,imgs)]
    dataset = monai.data.Dataset(data=data_dict,transform=data_transform)
    return dataset
def get_data_loader(config,dataset,target):
    batch_size = config['train']['batch_size']
    data_root = config['train']['data_root']
    num_classes = config['train']['num_classes']
    if dataset == 'feta':
        train_img = data_root+'/{}/train/img'.format(target)
        train_lab = data_root+'/{}/train/lab'.format(target)
        valid_img = data_root+'/{}/valid/img'.format(target)
        valid_lab = data_root+'/{}/valid/lab'.format(target)
        test_img = data_root+'/{}/test/img'.format(target)
        test_lab = data_root+'/{}/test/lab'.format(target)
    train_transform=Compose([
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        ScaleIntensityd(keys=["image"],minv=-1.0,maxv=1.0),
        SpatialPadd(keys=["image","label"],spatial_size=[32,64,64],mode='constant'),
        RandSpatialCropd(keys=["image","label"],roi_size=[32,64,64],random_size=False),
        AsDiscreted(keys=["label"],to_onehot=num_classes),
        ToTensord(keys=["image","label"])
    ])
    valid_transform=Compose([
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        ScaleIntensityd(keys=["image"],minv=-1.0,maxv=1.0),
        AsDiscreted(keys=["label"],to_onehot=num_classes),
        ToTensord(keys=["image","label"])
    ])
    train_set = get_datalist(train_img,train_lab,train_transform)
    valid_set = get_datalist(valid_img,valid_lab,valid_transform)
    test_set = get_datalist(test_img,test_lab,valid_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=1,shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=1,shuffle=False, drop_last=False)
    return train_loader,valid_loader,test_loader
def inference(input,model):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(32,64,64),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )
    return _compute(input)
def test(config,upl_model,valid_loader,test_loader,exp_name,dataset,target):
    device = torch.device('cuda:{}'.format(config['train']['gpu']))
    num_classes = config['train']['num_classes']
    for data_loader in [test_loader]:
        all_batch_dice = []
        all_batch_assd = []
        all_batch_hd = []
        output_result = []
        with torch.no_grad():
            for i, (data) in enumerate(data_loader):
                xt = data['image'].to(device)
                xt_label = data['label'].squeeze(0).to(device)
                output = inference(xt,upl_model)
                output = output.squeeze(0)  
                output = torch.argmax(output,dim=0)   
                xt_label = torch.argmax(xt_label,dim=0)    
                one_case_dice = dice_eval(output,xt_label,num_classes) * 100
                output_result.append(str(one_case_dice))
                all_batch_dice += [one_case_dice]
                one_case_assd = assd_eval(output,xt_label,num_classes)
                if config['test']['save_result']: 
                    name = data['name'][0]
                    results = "results_TMI/" + str(exp_name+'_'+target)
                    if(not os.path.exists(results)):
                            os.mkdir(results)
                    predict_dir  = os.path.join(results, name)
                    output = output.transpose(0,-1)
                    output_arr = output.cpu().numpy()
                    out_lab_obj = sitk.GetImageFromArray(output_arr/1.0)
                    sitk.WriteImage(out_lab_obj, predict_dir)
                output_result.append(str(one_case_assd))
                all_batch_assd += [one_case_assd]
        all_batch_dice = np.array(all_batch_dice)
        all_batch_assd = np.array(all_batch_assd)
        all_batch_hd = np.array(all_batch_hd)
        mean_dice = np.mean(all_batch_dice,axis=0) 
        std_dice = np.std(all_batch_dice,axis=0) 
        mean_assd = np.mean(all_batch_assd,axis=0)
        std_assd = np.std(all_batch_assd,axis=0)
        if dataset=='feta':
            print('{}±{} {}±{} {}±{} {}±{} {}±{} {}±{} {}±{}'.format(np.round(mean_dice[0],2),np.round(std_dice[0],2),np.round(mean_dice[1],2),np.round(std_dice[1],2),np.round(mean_dice[2],2),np.round(std_dice[2],2),np.round(mean_dice[3],2),np.round(std_dice[3],2),np.round(mean_dice[4],2),np.round(std_dice[4],2),np.round(mean_dice[5],2),np.round(std_dice[5],2),np.round(mean_dice[6],2),np.round(std_dice[6],2)))
            print('{}±{}'.format(np.round(np.mean(mean_dice,axis=0),2),np.round(np.mean(std_dice,axis=0),2)) )
            output_result.append('{}±{} {}±{} {}±{} {}±{} {}±{} {}±{} {}±{}'.format(np.round(mean_dice[0],2),np.round(std_dice[0],2),np.round(mean_dice[1],2),np.round(std_dice[1],2),np.round(mean_dice[2],2),np.round(std_dice[2],2),np.round(mean_dice[3],2),np.round(std_dice[3],2),np.round(mean_dice[4],2),np.round(std_dice[4],2),np.round(mean_dice[5],2),np.round(std_dice[5],2),np.round(mean_dice[6],2),np.round(std_dice[6],2)))
            output_result.append('{}±{}'.format(np.round(np.mean(mean_dice,axis=0),2),np.round(np.mean(std_dice,axis=0),2)) )
        if dataset=='feta':
            print('{}±{} {}±{} {}±{} {}±{} {}±{} {}±{} {}±{}'.format(np.round(mean_assd[0],2),np.round(std_assd[0],2),np.round(mean_assd[1],2),np.round(std_assd[1],2),np.round(mean_assd[2],2),np.round(std_assd[2],2),np.round(mean_assd[3],2),np.round(std_assd[3],2),np.round(mean_assd[4],2),np.round(std_assd[4],2),np.round(mean_assd[5],2),np.round(std_assd[5],2),np.round(mean_assd[6],2),np.round(std_assd[6],2)))
            print('{}±{}'.format(np.round(np.mean(mean_assd,axis=0),2),np.round(np.mean(std_assd,axis=0),2)) )
            output_result.append('{}±{} {}±{} {}±{} {}±{} {}±{} {}±{} {}±{}'.format(np.round(mean_assd[0],2),np.round(std_assd[0],2),np.round(mean_assd[1],2),np.round(std_assd[1],2),np.round(mean_assd[2],2),np.round(std_assd[2],2),np.round(mean_assd[3],2),np.round(std_assd[3],2),np.round(mean_assd[4],2),np.round(std_assd[4],2),np.round(mean_assd[5],2),np.round(std_assd[5],2),np.round(mean_assd[6],2),np.round(std_assd[6],2)))
            output_result.append('{}±{}'.format(np.round(np.mean(mean_assd,axis=0),2),np.round(np.mean(std_assd,axis=0),2)))
        with open('{}/result.txt'.format(results), 'w') as file:
            for line in output_result:
                file.write(line + "\n")
                
                
def train(config,train_loader,valid_loader,test_loader,target):
    # load exp_name
    exp_name = config['train']['exp_name']
    dataset = config['train']['dataset']
    if dataset=='fb':
        num_classes = config['network']['n_classes_fb']
    elif dataset=='mms':
        num_classes = config['network']['n_classes_mms']
    elif dataset=='feta':
        num_classes = config['network']['n_classes_feta']
    # load model
    device = torch.device('cuda:{}'.format(config['train']['gpu']))
    upl_model = UNet3d(config).to(device)
    upl_model.train()
    if target == 'd1':
        upl_model.load_state_dict(torch.load(config['train']['source_model_root'],map_location='cpu'),strict=False)
    else:
        raise "no such target modality"
    dec1 = upl_model.aux_dec1.state_dict()
    upl_model.aux_dec2.load_state_dict(dec1)
    upl_model.aux_dec3.load_state_dict(dec1)
    upl_model.aux_dec4.load_state_dict(dec1)
    # load train details
    num_epochs = config['train']['num_epochs']
    valid_epochs = config['train']['valid_epoch']
    best_dice = 0.

    for epoch in range(num_epochs):
        for i, (data) in enumerate(train_loader):
            B = data['image']
            B_label = data['label']
            B = B.to(device).detach()
            B_label = B_label.to(device).detach()
            if config['train']['train_target']:
                upl_model.save_nii(B)
                upl_model.trian_target(B)  
            else:
                upl_model.train_source(B,B_label) 
        # valid for target domain
        if (epoch) % valid_epochs == 0:
            current_dice = 0.
            with torch.no_grad():
                upl_model.eval()
                for it,(data) in enumerate(test_loader):  
                    xt = data['image'].to(device)
                    xt_label = data['label'].squeeze(0).to(device)
                    output = inference(xt,upl_model)
                    output = output.squeeze(0)  
                    output = torch.argmax(output,dim=0)   
                    xt_label = torch.argmax(xt_label,dim=0)   
                    one_case_dice = dice_eval(output,xt_label,num_classes) * 100
                    one_case_dice = np.array(one_case_dice)
                    one_case_dice = np.mean(one_case_dice,axis=0) 
                    current_dice += one_case_dice
            if (current_dice / (it+1)) > best_dice:
                best_dice = current_dice / (it+1)
                model_dir = "save_model_feta3d/" + str(exp_name+'_'+target)
                if(not os.path.exists(model_dir)):
                    os.mkdir(model_dir)
                best_epoch = '{}/model-{}-{}-{}.pth'.format(model_dir, 'best', str(epoch), np.round(best_dice,3))
                torch.save(upl_model.state_dict(), best_epoch)
                torch.save(upl_model.state_dict(), '{}/model-{}.pth'.format(model_dir, 'latest'))
            upl_model.train()

    upl_model.load_state_dict(torch.load(best_epoch,map_location='cpu'),strict=True)
    upl_model.eval()
    test(config,upl_model,valid_loader,test_loader,exp_name=exp_name,dataset=dataset,target=target)
    
    

def mian():
    # load config
    parser = argparse.ArgumentParser(description='config file')
    parser.add_argument('--config', type=str, default="./config/train3d.cfg",
                        help='Path to the configuration file')
    args = parser.parse_args()
    config = args.config
    config = parse_config(config)
    print(config)
    for dataset in ['feta']:
        for target in ['d1']:
            train_loader,valid_loader,test_loader = get_data_loader(config,dataset,target)
            train(config,train_loader,valid_loader,test_loader,target)
        
if __name__ == '__main__':
    set_random()
    mian()