## UPL-SFDA: Uncertainty-aware Pseudo Label Guided Source-Free Domain Adaptation for Medical Image Segmentation

This repository provides the code for "UPL-SFDA: Uncertainty-aware Pseudo Label Guided Source-Free Domain Adaptation for Medical Image Segmentation".

## Requirements
Non-exhaustive list:
* python3.6+
* Pytorch 1.8.1
* nibabel
* Scipy
* NumPy
* Scikit-image
* yaml
* tqdm
* pandas
* scikit-image
* SimpleITK

## Usage
1. Download the [Source model on M&MS, FB, and FeTA](https://drive.google.com/drive/folders/1WF0kwDBC_xchTG-oEWpbSQjKdHZdw80s?usp=drive_link) and move the extracted source model folder to the "save_model/source_model" directory in your project.
If you prefer, you can also train the source model yourself. To do this, navigate to the config directory and open the config\trainXX.cfg file. In the config file, locate the line that specifies train_target and change its value to False. 
For instance, you can train the source model using modality A on the M&MS datasets:
 ```
python train_source.py --config "./config/train2d_source.cfg"
```
2. Download the [M&MS Dataset](http://www.ub.edu/mnms), [FeTA Dataset](https://feta.grand-challenge.org/Data/), and organize the dataset directory structure as follows, for M&MS dataset:
```
your/M&MS_data_root/
       train/
            img/
                A/
                    A0S9V9_0.nii.gz
                    ...
                B/
                C/
                ...
            lab/
                A/
                    A0S9V9_0_gt.nii.gz
                    ...
                B/
                C/
                ...
       valid/
            img/
            lab/
       test/
           img/
           lab/
```
The network takes nii files as an input. The gt folder contains gray-scale images of the ground-truth, where the gray-scale level is the number of the class (0,1,...K).

3. Adaptation to the target domain, for 2D dataset:

```
python run_2d_upl.py --config "./config/train2d.cfg"
```
for 3D dataset:
```
python run_3d_upl.py --config "./config/train3d.cfg"
```