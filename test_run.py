
import numpy as np
import matplotlib
matplotlib.use('Agg')
import torch
from scipy import ndimage
import os
import SimpleTIK as sitk
from metrics import dice_eval,assd_eval

def test(config,upl_model,valid_loader,test_loader,list_data):
    dataset = config['train']['dataset']
    if dataset=='fb':
        num_classes = config['network']['n_classes_fb']
    elif dataset=='mms':
        num_classes = config['network']['n_classes_mms']
    elif dataset=='feta':
        num_classes = config['network']['n_classes_feta']
    device = torch.device('cuda:{}'.format(config['train']['gpu']))
    exp_name = config['train']['exp_name']
    for data_loader in [test_loader]:
        all_batch_dice = []
        all_batch_assd = []
        all_batch_hd = []
        with torch.no_grad():
            upl_model.train()
            for it,(xt,xt_label,xt_name,lab_Imag_dir) in enumerate(data_loader):  
                xt = xt.to(device)
                lab_x = xt_label.to(device)
                xt_label = xt_label.numpy().squeeze().astype(np.uint8)
                output = upl_model.test_with_name(xt)
                output = output.squeeze(0)
                output = torch.argmax(output,dim=1)        
                output_ = output.cpu().numpy()
                xt = xt.detach().cpu().numpy().squeeze()
                output = output_.squeeze()
                if config['test']['save_result']: 
                    lab_Imag = sitk.ReadImage(lab_Imag_dir[0])
                    lab_arr = sitk.GetArrayFromImage(lab_Imag)
                    output_ = np.expand_dims(output_,axis=0)
                    if len(lab_arr.shape) == 4:
                        e,a,b,c = lab_arr.shape
                    elif len(lab_arr.shape) == 3:
                        e,b,c = lab_arr.shape
                    ee,aa,bb,cc = output_.shape
                    zoom = [1,1,b/bb,c/cc]
                    output_ = ndimage.zoom(output_, zoom,order=0)
                    output_ = output_.squeeze(0).astype(np.float)
                    name = str(xt_name)[2:-3]
                    results = "results/" + str(exp_name)
                    if(not os.path.exists(results)):
                            os.mkdir(results)
                    predict_dir  = os.path.join(results, name)
                    out_lab_obj = sitk.GetImageFromArray(output_)
                    out_lab_obj.CopyInformation(lab_Imag)
                    sitk.WriteImage(out_lab_obj, predict_dir)
                lab_Imag = sitk.ReadImage(lab_Imag_dir)
                lab_arr = sitk.GetArrayFromImage(lab_Imag)
                e,a,b,c = lab_arr.shape
                ee,bb,cc = output.shape
                zoom = [1,b/bb,c/cc]
                output = ndimage.zoom(output, zoom,order=0)
                xt_label = ndimage.zoom(xt_label, zoom,order=0)
                one_case_dice = dice_eval(output,xt_label,num_classes) * 100
                all_batch_dice += [one_case_dice]
                one_case_assd = assd_eval(output,xt_label,num_classes)
                all_batch_assd.append(one_case_assd)
                # all_batch_hd.append(one_case_hd95)
        all_batch_dice = np.array(all_batch_dice)
        all_batch_assd = np.array(all_batch_assd)
        mean_dice = np.mean(all_batch_dice,axis=0) 
        std_dice = np.std(all_batch_dice,axis=0) 
        mean_assd = np.mean(all_batch_assd,axis=0)
        std_assd = np.std(all_batch_assd,axis=0)
        print(mean_dice,std_dice,mean_assd,std_assd)
        if dataset=='mms':
            print('{}±{} {}±{} {}±{}'.format(np.round(mean_dice[0],2),np.round(std_dice[0],2),np.round(mean_dice[1],2),np.round(std_dice[1],2),np.round(mean_dice[2],2),np.round(std_dice[2],2)))
            print('{}±{}'.format(np.round(np.mean(mean_dice,axis=0),2),np.round(np.mean(std_dice,axis=0),2)) )
            list_data.append('{}±{} {}±{} {}±{}'.format(np.round(mean_dice[0],2),np.round(std_dice[0],2),np.round(mean_dice[1],2),np.round(std_dice[1],2),np.round(mean_dice[2],2),np.round(std_dice[2],2)))
            list_data.append('{}±{}'.format(np.round(np.mean(mean_dice,axis=0),2),np.round(np.mean(std_dice,axis=0),2)) )
        elif dataset=='fb':
            print('{}±{}'.format(np.round(mean_dice[0],2),np.round(std_dice[0],2)))
            list_data.append('{}±{}'.format(np.round(mean_dice[0],2),np.round(std_dice[0],2)))
        if dataset=='mms':
            # print('ASSD:')
            print('{}±{} {}±{} {}±{}'.format(np.round(mean_assd[0],2),np.round(std_assd[0],2),np.round(mean_assd[1],2),np.round(std_assd[1],2),np.round(mean_assd[2],2),np.round(std_assd[2],2)))
            print('{}±{}'.format(np.round(np.mean(mean_assd,axis=0),2),np.round(np.mean(std_assd,axis=0),2)) )
            list_data.append('{}±{} {}±{} {}±{}'.format(np.round(mean_assd[0],2),np.round(std_assd[0],2),np.round(mean_assd[1],2),np.round(std_assd[1],2),np.round(mean_assd[2],2),np.round(std_assd[2],2)))
            list_data.append('{}±{}'.format(np.round(np.mean(mean_assd,axis=0),2),np.round(np.mean(std_assd,axis=0),2)) )
        elif dataset=='fb':
            print('{}±{}'.format(np.round(mean_assd[0],2),np.round(std_assd[0],2)))
            list_data.append('{}±{}'.format(np.round(mean_assd[0],2),np.round(std_assd[0],2)))
    return list_data


    

    
