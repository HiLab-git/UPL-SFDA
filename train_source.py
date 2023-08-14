from utils import parse_config, set_random,niiDataset
from unet import UNet
from torch.utils.data import DataLoader
import torch
import matplotlib
import os
import argparse
from test_run import test
from metrics import dice_eval
import numpy as np

matplotlib.use('Agg')
def get_data_loader(config,dataset,target):
    batch_size = config['train']['batch_size']
    data_root_mms = config['train']['data_root_mms']
    data_root_fb = config['train']['data_root_fb']
    if dataset == 'mms':
        train_img = data_root_mms+'/train/img/{}'.format(target)
        train_lab = data_root_mms+'/train/lab/{}'.format(target)
        valid_img = data_root_mms+'/valid/img/{}'.format(target)
        valid_lab = data_root_mms+'/valid/lab/{}'.format(target)
        test_img = data_root_mms+'/test/img/{}'.format(target)
        test_lab = data_root_mms+'/test/lab/{}'.format(target)
    elif dataset == 'fb':
        train_img = data_root_fb+'/{}/image/train'.format(target)
        train_lab = data_root_fb+'/{}/label/train'.format(target)
        valid_img = data_root_fb+'/{}/image/valid'.format(target)
        valid_lab = data_root_fb+'/{}/label/valid'.format(target)
        test_img = data_root_fb+'/{}/image/test'.format(target)
        test_lab = data_root_fb+'/{}/label/test'.format(target)
    
    train_test = niiDataset(train_img,train_lab, dataset=dataset, target = target, phase = 'train')
    train_loader = DataLoader(train_test, batch_size = batch_size,shuffle=True, drop_last=True)
    val_dataset = niiDataset(valid_img,valid_lab, dataset=dataset, target = target, phase = 'valid')
    valid_loader = DataLoader(val_dataset, batch_size=1,shuffle=False, drop_last=False)
    test_dataset = niiDataset(test_img,test_lab, dataset=dataset, target = target, phase = 'test')
    test_loader = DataLoader(test_dataset, batch_size=1,shuffle=False, drop_last=False)
    return train_loader,valid_loader,test_loader

def train(config,train_loader,valid_loader,test_loader,target,list_data):
    # load exp_name
    exp_name = config['train']['exp_name']
    dataset = config['train']['dataset']
    if dataset=='fb':
        num_classes = config['network']['n_classes_fb']
    elif dataset=='mms':
        num_classes = config['network']['n_classes_mms']
    # load model
    device = torch.device('cuda:{}'.format(config['train']['gpu']))
    upl_model = UNet(config).to(device)
    upl_model.train()
    upl_model.initialize()
    print("model initialize")
    # load train details
    num_epochs = config['train']['num_epochs']
    valid_epochs = config['train']['valid_epoch']
    j = 0
    best_dice = 0.
    for epoch in range(num_epochs):
        for i, (B, B_label, _,_) in enumerate(train_loader):
            B = B.to(device).detach()
            B_label = B_label.to(device).detach()
            upl_model.train_source(B,B_label) 
        if (epoch) % valid_epochs == 0:
            current_dice = 0.
            with torch.no_grad():
                upl_model.eval()
                for it,(xt,xt_label,xt_name,lab_Imag) in enumerate(valid_loader):  
                    xt = xt.to(device)
                    xt_label = xt_label.numpy().squeeze().astype(np.uint8)
                    output,_ = upl_model.test_with_name(xt,xt_name)
                    output = output.squeeze(0)
                    output = torch.argmax(output,dim=1)        
                    output_ = output.cpu().numpy()
                    xt = xt.detach().cpu().numpy().squeeze()
                    output = output_.squeeze()
                    one_case_dice = dice_eval(output,xt_label,num_classes) * 100
                    one_case_dice = np.array(one_case_dice)
                    one_case_dice = np.mean(one_case_dice,axis=0) 
                    current_dice += one_case_dice
            if (current_dice / (it+1)) > best_dice:
                best_dice = current_dice / (it+1)
                model_dir = "save_model_revised_TMI/" + str(exp_name+'_'+target)
                if(not os.path.exists(model_dir)):
                    os.mkdir(model_dir)
                best_epoch = '{}/model-{}-{}-{}.pth'.format(model_dir, 'best', str(j+1), best_dice)
                torch.save(upl_model.state_dict(), best_epoch)
                torch.save(upl_model.state_dict(), '{}/model-{}.pth'.format(model_dir, 'latest'))
        upl_model.update_lr()
    upl_model.load_state_dict(torch.load(best_epoch,map_location='cpu'),strict=False)
    upl_model.eval()
    test(config,upl_model,valid_loader,test_loader,list_data)
    return list_data
    
    

def mian():
    # load config
    # load config
    parser = argparse.ArgumentParser(description='config file')
    parser.add_argument('--config', type=str, default="./config\train2d.cfg",
                        help='Path to the configuration file')
    args = parser.parse_args()
    config = args.config
    config = parse_config(config)
    list_data = []
    print(config)
    dataset = config['train']['dataset']
    for dataset in ['mms','fb']:
        if dataset == 'mms':
            for target in ['A']:
                config['train']['dataset'] = dataset
                list_data.append(dataset)
                list_data.append(target)
                train_loader,valid_loader,test_loader = get_data_loader(config,dataset,target)
                list_data = train(config,train_loader,valid_loader,test_loader,target,list_data)
        elif dataset == 'fb':
            for target in ['source']:
                config['train']['dataset'] = dataset
                list_data.append(dataset)
                list_data.append(target)
                train_loader,valid_loader,test_loader = get_data_loader(config,dataset,target)
                list_data = train(config,train_loader,valid_loader,test_loader,target,list_data)
    with open('result_data/source_wo_en.txt', 'w') as file:
        for line in list_data:
            file.write(line + "\n")
        
if __name__ == '__main__':
    set_random()
    mian()