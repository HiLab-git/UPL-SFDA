##load data
from torch.utils.data import Dataset
import numpy as np
import os
import torch
import yaml
import random
import configparser
import math
from torch.nn import init
from scipy import ndimage
import SimpleITK as sitk 
import torch.nn as nn


def is_int(val_str):
    start_digit = 0
    if(val_str[0] =='-'):
        start_digit = 1
    flag = True
    for i in range(start_digit, len(val_str)):
        if(str(val_str[i]) < '0' or str(val_str[i]) > '9'):
            flag = False
            break
    return flag

def is_float(val_str):
    flag = False
    if('.' in val_str and len(val_str.split('.'))==2 and not('./' in val_str)):
        if(is_int(val_str.split('.')[0]) and is_int(val_str.split('.')[1])):
            flag = True
        else:
            flag = False
    elif('e' in val_str and val_str[0] != 'e' and len(val_str.split('e'))==2):
        if(is_int(val_str.split('e')[0]) and is_int(val_str.split('e')[1])):
            flag = True
        else:
            flag = False       
    else:
        flag = False
    return flag 

def is_bool(var_str):
    if( var_str.lower() =='true' or var_str.lower() == 'false'):
        return True
    else:
        return False
    
def parse_bool(var_str):
    if(var_str.lower() =='true'):
        return True
    else:
        return False
     
def is_list(val_str):
    if(val_str[0] == '[' and val_str[-1] == ']'):
        return True
    else:
        return False

def parse_list(val_str):
    sub_str = val_str[1:-1]
    splits = sub_str.split(',')
    output = []
    for item in splits:
        item = item.strip()
        if(is_int(item)):
            output.append(int(item))
        elif(is_float(item)):
            output.append(float(item))
        elif(is_bool(item)):
            output.append(parse_bool(item))
        elif(item.lower() == 'none'):
            output.append(None)
        else:
            output.append(item)
    return output
    
def parse_value_from_string(val_str):
    if(is_int(val_str)):
        val = int(val_str)
    elif(is_float(val_str)):
        val = float(val_str)
    elif(is_list(val_str)):
        val = parse_list(val_str)
    elif(is_bool(val_str)):
        val = parse_bool(val_str)
    elif(val_str.lower() == 'none'):
        val = None
    else:
        val = val_str
    return val

def parse_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    output = {}
    for section in config.sections():
        output[section] = {}
        for key in config[section]:
            val_str = str(config[section][key])
            if(len(val_str)>0):
                val = parse_value_from_string(val_str)
                output[section][key] = val
            else:
                val = None
            print(section, key, val_str, val)
    return output


def load_npz(path):
    img = np.load(path)['arr_0']
    gt = np.load(path)['arr_1']
    return img, gt
    
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream,Loader=yaml.FullLoader)

def set_random(seed_id=1234):
    np.random.seed(seed_id)
    torch.manual_seed(seed_id)   #for cpu
    torch.cuda.manual_seed_all(seed_id) #for GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# config setting
def is_int(val_str):
    start_digit = 0
    if(val_str[0] =='-'):
        start_digit = 1
    flag = True
    for i in range(start_digit, len(val_str)):
        if(str(val_str[i]) < '0' or str(val_str[i]) > '9'):
            flag = False
            break
    return flag

def is_float(val_str):
    flag = False
    if('.' in val_str and len(val_str.split('.'))==2 and not('./' in val_str)):
        if(is_int(val_str.split('.')[0]) and is_int(val_str.split('.')[1])):
            flag = True
        else:
            flag = False
    elif('e' in val_str and val_str[0] != 'e' and len(val_str.split('e'))==2):
        if(is_int(val_str.split('e')[0]) and is_int(val_str.split('e')[1])):
            flag = True
        else:
            flag = False       
    else:
        flag = False
    return flag 

def is_bool(var_str):
    if( var_str.lower() =='true' or var_str.lower() == 'false'):
        return True
    else:
        return False
    
def parse_bool(var_str):
    if(var_str.lower() =='true'):
        return True
    else:
        return False
     
def is_list(val_str):
    if(val_str[0] == '[' and val_str[-1] == ']'):
        return True
    else:
        return False

def parse_list(val_str):
    sub_str = val_str[1:-1]
    splits = sub_str.split(',')
    output = []
    for item in splits:
        item = item.strip()
        if(is_int(item)):
            output.append(int(item))
        elif(is_float(item)):
            output.append(float(item))
        elif(is_bool(item)):
            output.append(parse_bool(item))
        elif(item.lower() == 'none'):
            output.append(None)
        else:
            output.append(item)
    return output

def parse_value_from_string(val_str):
    if(is_int(val_str)):
        val = int(val_str)
    elif(is_float(val_str)):
        val = float(val_str)
    elif(is_list(val_str)):
        val = parse_list(val_str)
    elif(is_bool(val_str)):
        val = parse_bool(val_str)
    elif(val_str.lower() == 'none'):
        val = None
    else:
        val = val_str
    return val

def parse_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    output = {}
    for section in config.sections():
        output[section] = {}
        for key in config[section]:
            val_str = str(config[section][key])
            if(len(val_str)>0):
                val = parse_value_from_string(val_str)
                output[section][key] = val
            else:
                val = None
            print(section, key, val_str, val)
    return output

class UnpairedDataset(Dataset):
    #get unpaired dataset, such as MR-CT dataset
    def __init__(self,A_path,B_path):
        listA = os.listdir(A_path)
        listB = os.listdir(B_path)
        self.listA = [os.path.join(A_path,k) for k in listA]
        self.listB = [os.path.join(B_path,k) for k in listB]
        self.Asize = len(self.listA)
        self.Bsize = len(self.listB)
        self.dataset_size = max(self.Asize,self.Bsize)
        
    def __getitem__(self,index):
        if self.Asize == self.dataset_size:
            A,A_gt = load_npz(self.listA[index])
            B,B_gt = load_npz(self.listB[random.randint(0, self.Bsize - 1)])
        else :
            B,B_gt = load_npz(self.listB[index])
            A,A_gt = load_npz(self.listA[random.randint(0, self.Asize - 1)])


        A = torch.from_numpy(A.copy()).unsqueeze(0).float()
        A_gt = torch.from_numpy(A_gt.copy()).unsqueeze(0).float()
        B = torch.from_numpy(B.copy()).unsqueeze(0).float()
        B_gt = torch.from_numpy(B_gt.copy()).unsqueeze(0).float()
        return A,A_gt,B,B_gt
        
    def __len__(self):
        return self.dataset_size

def crop_depth(img,lab,phase = 'train'):
    D,H,W = img.shape
    if D > 10:
        if phase == 'train':
            target_ssh = np.random.randint(0, int(D-10), 1)[0]
            zero_img = img[target_ssh:target_ssh+10,:,:]
            zero_lab = lab[target_ssh:target_ssh+10,:,:]
        elif phase == 'valid':
            zero_img,zero_lab = img,lab
        elif phase == 'feta':
            sample_indices = np.random.choice(D, size=10, replace=False)
            zero_img = np.zeros((10,H,W))
            zero_lab = np.zeros((10,H,W))
            for i, index in enumerate(sample_indices):
                zero_img[i] = img[index]
                zero_lab[i] = lab[index]
    else:
        zero_img = np.zeros((10,H,W))
        zero_lab = np.zeros((10,H,W))
        zero_img[0:D,:,:] = img
        zero_lab[0:D,:,:] = lab
    return zero_img,zero_lab

def winadj_mri(array):
    v0 = np.percentile(array, 1)
    v1 = np.percentile(array, 99)
    array[array < v0] = v0    
    array[array > v1] = v1  
    v0 = array.min() 
    v1 = array.max() 
    array = (array - v0) / (v1 - v0) * 2.0 - 1.0
    return array

def resize(img,lab):
    D,H,W = img.shape
    zoom = [1,256/H,256/W]
    img=ndimage.zoom(img,zoom,order=2)
    lab=ndimage.zoom(lab,zoom,order=0)
    return img,lab

class niiDataset(Dataset):
    def __init__(self, source_img,source_lab,dataset,target, phase = 'test'):
        self.dataset = dataset
        self.source_img = source_img
        self.source_lab = source_lab
        self.phase  =phase
        nii_names = os.listdir(source_img)
        self.all_files = []
        for nii_name in nii_names:
            self.img_path = os.path.join(self.source_img, nii_name)
            if self.dataset == 'fb' or self.dataset == 'feta':
                self.lab_path = os.path.join(self.source_lab, nii_name[:-7] + '_seg.nii.gz')
            elif self.dataset == 'mms':
                self.lab_path = os.path.join(self.source_lab, nii_name[:-7] + '_gt.nii.gz')
            else:
                print(self.dataset)
                raise Exception('Unrecognized dataset.')
            
            self.nii_name = str(nii_name)
            self.all_files.append({
                "img": self.img_path,
                "lab": self.lab_path,
                "img_name": self.nii_name
            })    
    def __getitem__(self, index):
        fname = self.all_files[index]
        img_obj = sitk.ReadImage(fname["img"])
        A = sitk.GetArrayFromImage(img_obj) /1 
        lab_obj = sitk.ReadImage(fname["lab"])
        A_gt = sitk.GetArrayFromImage(lab_obj)
        if self.phase == 'train' and self.dataset == 'mms':
            A,A_gt = crop_depth(A,A_gt)
        if self.dataset == 'fb':
            if self.phase == 'train':
                A,A_gt = crop_depth(A,A_gt,phase = 'train')
            elif self.phase == 'valid':
                A,A_gt = crop_depth(A,A_gt,phase = 'valid')
            A = winadj_mri(A)
            A,A_gt = resize(A,A_gt)
        elif self.dataset == 'feta':
            if self.phase == 'train':
                A,A_gt = crop_depth(A,A_gt,phase = 'feta')
        A = winadj_mri(A)
        A,A_gt = resize(A,A_gt)
        A = torch.from_numpy(A.copy()).unsqueeze(0).float()
        A_gt = torch.from_numpy(A_gt.copy()).unsqueeze(0).float()
        A,A_gt = random_rotate(A,A_gt)
        return A, A_gt, fname["img_name"],fname["lab"]  
           
    def __len__(self):
        return len(self.all_files)


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
def one_hot_encode(input_tensor):
    if len(input_tensor.shape) == 4:
        a,b,c,d = input_tensor.shape
    elif len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(1)
    tensor_list = []
    for i in range(4):
        tmp = (input_tensor==i) * torch.ones_like(input_tensor)
        tensor_list.append(tmp)
    output_tensor = torch.cat(tensor_list,dim=1)
    return output_tensor.float()

def get_largest_component(image):
    dim = len(image.shape)
    if(image.sum() == 0 ):
        print('the largest component is null')
        return image
    if(dim == 2):
        s = ndimage.generate_binary_structure(2,1)
    elif(dim == 3):
        s = ndimage.generate_binary_structure(3,1)
    else:
        raise ValueError("the dimension number should be 2 or 3")
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    output = np.asarray(labeled_array == max_label[0], np.uint8)
    return output


def tensor_rot_90(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.flip(3).transpose(2, 3)
    else:
	    return x.flip(2).transpose(1, 2)
def tensor_rot_180(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.flip(3).flip(2)
    else:
	    return x.flip(2).flip(1)
def tensor_flip_2(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.flip(2)
    else:
	    return x.flip(1)
def tensor_flip_3(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.flip(3)
    else:
	    return x.flip(2)

def tensor_rot_270(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.transpose(2, 3).flip(3)
    else:
        return x.transpose(1, 2).flip(2)
    
def rotate_single_random(img):
    x_shape = list(img.shape)
    if(len(x_shape) == 5):
        [N, C, D, H, W] = x_shape
        new_shape = [N*D, C, H, W]
        x = torch.transpose(img, 1, 2)
        img = torch.reshape(x, new_shape)
    label = np.random.randint(0, 4, 1)[0]
    if label == 1:
        img = tensor_rot_90(img)
    elif label == 2:
        img = tensor_rot_180(img)
    elif label == 3:
        img = tensor_rot_270(img)
    else:
        img = img
    return img,label

def rotate_single_with_label(img, label):
    if label == 1:
        img = tensor_rot_90(img)
    elif label == 2:
        img = tensor_rot_180(img)
    elif label == 3:
        img = tensor_rot_270(img)
    else:
        img = img
    return img

def random_rotate(A,A_gt):
    target_ssh = np.random.randint(0, 8, 1)[0]
    A = rotate_single_with_label(A, target_ssh)
    A_gt = rotate_single_with_label(A_gt, target_ssh)
    return A,A_gt

def rotate_4(img):
    # target_ssh = np.random.randint(0, 4, 1)[0]
    A_1 = rotate_single_with_label(img, 1)
    A_2 = rotate_single_with_label(img, 2)
    A_3 = rotate_single_with_label(img, 3)
    return A_1,A_2,A_3


