import numpy as np
import SimpleITK as sitk
import os
import glob
import random

feta_root_ = '/data2/jianghao/data/fetal_brain/feta_2.2'
feta_root = '/data2/jianghao/data/fetal_brain/feta_2.2/proce'
sub_folds = glob.glob(feta_root_ + '/**/*sub*/', recursive=True)
# print(sub_folds,len(sub_folds))
# random.shuffle(sub_folds)
sub_folds.sort()
d1_set = []
d2_set = []
for sub_name in sub_folds:
    uid = sub_name.split('/')[-2]
    uid = int(uid.split('-')[-1])
    print('num_uid:',uid)
    if uid <41:
        d1_set.append(sub_name)
    elif uid >40:
        d2_set.append(sub_name)
random.shuffle(d1_set)
random.shuffle(d2_set)
       

for aa in [1, 2]:
    if aa == 1:
        sub_foldsss = d1_set
    elif aa == 2:
        sub_foldsss = d2_set
    for j in range(len(sub_foldsss)):
        i = j+1
        name = sub_foldsss[j].split('/')[-2]
        sub_fold = os.path.join(sub_foldsss[j],'anat')
        names = os.listdir(sub_fold)
        lab = [item for item in names if 'seg.nii.gz' in item][0]
        img = [item for item in names if 'T2w.nii.gz' in item][0]
        lab_ = sitk.ReadImage(os.path.join(sub_fold,lab))
        lab = sitk.GetArrayFromImage(lab_)
        spacing = lab_.GetSpacing()
        
        
        img_ = sitk.ReadImage(os.path.join(sub_fold,img))
        img = sitk.GetArrayFromImage(img_)

        indi = np.where(lab > 0)
        d0,d1 = indi[0].min(),indi[0].max()
        w0,w1 = indi[1].min(),indi[1].max()
        h0,h1 = indi[2].min(),indi[2].max()
        lab = lab[d0:d1,max(0,w0-10):(w1+10),max(0,h0-10):(h1+10)]
        img = img[d0:d1,max(0,w0-10):(w1+10),max(0,h0-10):(h1+10)]
        assert lab.shape == img.shape

        img_p = sitk.GetImageFromArray(img)
        img_p.SetSpacing(spacing)
        lab_p = sitk.GetImageFromArray(lab)
        lab_p.SetSpacing(spacing)
        
        if aa == 1:
            set = 'd1'
        if aa == 2:
            set = 'd2'
        if i <= 28:
            split = 'train'
        elif i <= 32 and i>28:
            split = 'valid'
        elif i <= 40 and i>32:
            split = 'test'  
        sitk.WriteImage(lab_p,os.path.join(feta_root,set,split,'lab',name+'_seg.nii.gz'))
        sitk.WriteImage(img_p,os.path.join(feta_root,set,split,'img',name+'.nii.gz'))
