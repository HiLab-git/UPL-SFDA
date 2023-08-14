import numpy as np
import SimpleITK as sitk
import os
from glob import glob
import pandas as pd
from medpy import metric
from skimage.morphology import skeletonize_3d
from tqdm import tqdm
from scipy.ndimage import zoom

def postprocessing_sitk(label_img, spacing, ignore_label=[7, 8, 11], fillhole_label=[1, 5], remove_label=[2, 3, 5, 9, 10], remain_two=[], gallbladder_label=4, remove_percent=0.2, remove_volume=300):
    labels = np.unique(label_img)
    sitk_label = sitk.GetImageFromArray(label_img)
    for label in labels:
        if label == 0 or label in ignore_label:
            continue
        this_sitk_label = sitk_label == label

        if label in fillhole_label:
            this_sitk_label = sitk.BinaryFillhole(this_sitk_label)

        cc = sitk.ConnectedComponent(this_sitk_label)
        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.SetGlobalDefaultNumberOfThreads(8)
        stats.Execute(cc, this_sitk_label)

        props = [(stats.GetPhysicalSize(l), l) for l in stats.GetLabels()]
        props = sorted(props, key=lambda x: x[0], reverse=True)

        if len(props) > 1:
            cc_np = sitk.GetArrayFromImage(cc)
            if label in remove_label:
                whole_area = np.sum([area for area, _ in props])
                for area, label_id in props[1:]:
                    if area < remove_percent * whole_area:
                        label_img[cc_np == label_id] = 0
            elif label in remain_two:
                if len(props) > 2:
                    for _, label_id in props[2:]:
                        label_img[cc_np == label_id] = 0
            else:
                for _, label_id in props[1:]:
                    label_img[cc_np == label_id] = 0

        elif label in fillhole_label or label == gallbladder_label:
            cc_np = sitk.GetArrayFromImage(cc)

        if label == gallbladder_label and props[0][0] * spacing[0] * spacing[1] * spacing[2] < remove_volume:
            label_img[cc_np == props[0][1]] = 0

        if label in fillhole_label:
            label_img[cc_np == props[0][1]] = label

    kidney_volume = np.sum(label_img == 2) + np.sum(label_img == 3)
    for label in [2, 3]:
        if np.sum(label_img == label) <= 0.15 * kidney_volume:
            label_img[label_img == label] = 0
    return label_img.astype(np.uint8)

def image_resize(img, origin_spacing, target_spacing=[1, 1, 1], order=0):
    scale = np.array(origin_spacing) / np.array(target_spacing)
    img = zoom(img, zoom=list(scale), order=order)
    if order == 0:
        img = img.astype(np.uint8)
    else:
        img = img.astype(np.float)
    return img


def clDice(v_p, v_l):
    def cl_score(v, s):
        return np.sum(v * s) / np.sum(s)
    tprec = cl_score(v_p, skeletonize_3d(v_l))
    tsens = cl_score(v_l, skeletonize_3d(v_p))
    return (2*tprec*tsens + 1e-8) / (tprec + tsens + 1e-8)

def Dice(v_p, v_l, smooth=1e-5):
    v_p = v_p.astype(np.bool).reshape(-1)
    v_l = v_l.astype(np.bool).reshape(-1)
    intersection = np.sum(v_p * v_l)
    return (2.0 * intersection + smooth) / (v_p.sum() + v_l.sum() + smooth)

def calc_metrics(type, predmask, gtmask):
    try:
        if type == "hd":
            return metric.binary.hd95(predmask, gtmask)
        elif type == "asd":
            return metric.binary.asd(predmask, gtmask)
        elif type == 'ravd':
            return metric.binary.ravd(predmask, gtmask)
        elif type == "dc":
            return Dice(predmask, gtmask)
            # return metric.binary.dc(predmask, gtmask)
        elif type == "jc":
            return metric.binary.jc(predmask, gtmask)
    except:
        return 0

def cal_metrics_multi_class(pred, gt, cls_num, partial_label=None):
    cls_hds = []
    cls_asds = []
    cls_dcs = []
    cls_ravds = []

    for i in range(1, cls_num):
        if (partial_label is not None and partial_label[i-1] == 0) or i not in [2, 3]:
            cls_dcs.append(np.nan)
            cls_ravds.append(np.nan)
            cls_hds.append(np.nan)
            cls_asds.append(np.nan)
            continue

        pred_mask = (pred == i).astype(np.uint8)
        gt_mask = (gt == i).astype(np.uint8)

        try:
            this_dc = calc_metrics('dc', pred_mask, gt_mask)
        except Exception as ex:
            print('exception in calculating Dice: {}'.format(ex))
            this_dc = np.nan
        # try:
        #     this_ravd = calc_metrics('ravd', pred_mask, gt_mask)
        # except Exception as ex:
        #     print('exception in calculating RAVD: {}'.format(ex))
        #     this_ravd = np.nan
        try:
            this_hd = calc_metrics('hd', pred_mask, gt_mask)
        except Exception as ex:
            print('exception in calculating HD95: {}'.format(ex))
            this_hd = np.nan
        # try:
        #     this_asd = calc_metrics('asd', pred_mask, gt_mask)
        # except Exception as ex:
        #     print('exception in calculating ASD: {}'.format(ex))
        #     this_asd = np.nan

        # this_dc = np.nan
        this_ravd = np.nan
        # this_hd = np.nan
        this_asd = np.nan

        cls_dcs.append(this_dc)
        cls_ravds.append(this_ravd)
        cls_hds.append(this_hd)
        cls_asds.append(this_asd)

    return cls_dcs, cls_ravds, cls_hds, cls_asds

def calc_mean_std(arr):
    arr = [i for i in arr if not np.isnan(i)]
    return np.mean(arr), np.std(arr)

def cal(np_pred,np_label,resize2unify):
    

    cls_hd95s = []
    cls_asds = []
    cls_dcs = []
    cls_ravds = []
    sample_names = []

    if resize2unify:
        # print(sitk_pred.GetSpacing()[::-1], sitk_label.GetSpacing()[::-1])
        np_pred = image_resize(np_pred, sitk_pred.GetSpacing()[::-1])
        np_label = image_resize(np_label, sitk_label.GetSpacing()[::-1])

    flag = (np.max(np.unique(np_pred)) == 1)
    
    partial_label = [1] * 4

    this_cls_dc, this_cls_ravd, this_cls_hd95, this_cls_asd = cal_metrics_multi_class(np_pred, np_label, len(partial_label) + 1, partial_label)

    cls_dcs.append(this_cls_dc)
    cls_ravds.append(this_cls_ravd)
    cls_hd95s.append(this_cls_hd95)
    cls_asds.append(this_cls_asd)
    sample_names.append(name.split('.nii.gz')[0])

    cls_dcs = np.array(cls_dcs)
    cls_ravds = np.array(cls_ravds)
    cls_hd95s = np.array(cls_hd95s)
    cls_asds = np.array(cls_asds)

    if flag:
        class_names = ['liver']
    else:
        class_names = ['spleen', 'R-kidney', 'L-kidney', 'gallbladder', 'liver', 'stomach', 'artery', 'IVC', 'duodenum', 'pancrease', 'esophagus']

    print('{:^15}\t{:^15}\t{:^15}\t{:^15}\t{:^15}'.format('name', 'dice', 'ravd', 'hd95', 'asd'))
    p_format = '{:^15}\t{:>7.4f}({:<6.4f})\t{:>7.4f}({:<6.4f})\t{:>7.4f}({:<6.4f})\t{:>7.4f}({:<6.4f})'
    mean_dice = []
    mean_ravd = []
    mean_hd95 = []
    mean_asd = []

    for cls_no, cls in enumerate(class_names):
        dc, dc_std = calc_mean_std(cls_dcs[:, cls_no])
        ravd, ravd_std = calc_mean_std(cls_ravds[:, cls_no])
        hd95, hd95_std = calc_mean_std(cls_hd95s[:, cls_no])
        asd, asd_std = calc_mean_std(cls_asds[:, cls_no])
        print(p_format.format(cls, dc, dc_std, ravd, ravd_std, hd95, hd95_std, asd, asd_std))
        mean_dice.append((dc, dc_std))
        mean_ravd.append((ravd, ravd_std))
        mean_hd95.append((hd95, hd95_std))
        mean_asd.append((asd, asd_std))

    column_names = ['name'] + [cls_name+'_dice' for cls_name in class_names] + [cls_name+'_ravd' for cls_name in class_names] + [cls_name+'_hd95' for cls_name in class_names] + [cls_name+'_asd' for cls_name in class_names]

    # 个例分析
    metrics_df = pd.DataFrame(columns=column_names)
    for this_name, this_dc, this_ravd, this_hd95, this_asd in zip(sample_names, cls_dcs, cls_ravds, cls_hd95s, cls_asds):
        row_data = [this_name,] + list(this_dc) + list(this_ravd) + list(this_hd95) + list(this_asd)
        row_data = pd.DataFrame(np.array(row_data).reshape(1, -1), columns=column_names)
        metrics_df = metrics_df.append(row_data)

    metrics_df.to_csv(os.path.join(save_path, 'metrics_per_case_hdmm.csv'), index=False)

    # 汇总分析
    column_names = [''] + list(class_names)
    metrics_df = pd.DataFrame(columns=column_names)
    for metric_name, metric in zip(['Dice', 'RAVD', 'HD95', 'ASD'], [mean_dice, mean_ravd, mean_hd95, mean_asd]):
        row_data = [metric_name] + ['{:.4f}({:.4f})'.format(x[0], x[1]) for x in metric]
        row_data = pd.DataFrame(np.array(row_data).reshape(1, -1), columns=column_names)

        metrics_df = metrics_df.append(row_data)
    metrics_df.to_csv(os.path.join(save_path, 'metrics_hdmm_kidney.csv'), index=False)


if __name__ == '__main__':
    main()



