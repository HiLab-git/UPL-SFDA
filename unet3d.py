from scipy import ndimage
import torch
import torch.nn as nn
import numpy as np
from utils import init_weights, rotate_single_with_label
from torch.optim import lr_scheduler
from torch.nn import init
import monai.losses as losses

dice_loss = losses.DiceLoss()
def get_scheduler(optimizer):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
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

def get_largest_component(image):
    """
    get the largest component from 2D or 3D binary image
    image: nd array
    """
    dim = len(image.shape)
    # print(dim,image.shape)
    if(image.sum() == 0 ):
        # print('the largest component is null')
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
    output = np.asarray(labeled_array == max_label, np.uint8)
    return  output

class ConvBlock(nn.Module):
    """
    Two 3D convolution layers with batch norm and leaky relu.
    Droput is used between the two convolution layers.
    
    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    """
    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU()
        )
       
    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """
    3D downsampling followed by ConvBlock

    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    """
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class UpBlock(nn.Module):
    """
    3D upsampling followed by ConvBlock
    
    :param in_channels1: (int) Channel number of high-level features.
    :param in_channels2: (int) Channel number of low-level features.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    :param trilinear: (bool) Use trilinear for up-sampling (by default).
        If False, deconvolution is used for up-sampling. 
    """
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 trilinear=True):
        super(UpBlock, self).__init__()
        self.trilinear = trilinear
        if trilinear:
            self.conv1x1 = nn.Conv3d(in_channels1, in_channels2, kernel_size = 1)
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.trilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class Encoder(nn.Module):
    def __init__(self,
        in_chns,
        n_classes,
        ft_chns,
        dropout_p
        ):
        super().__init__()
        self.in_chns   = in_chns
        self.ft_chns   = ft_chns
        self.n_class   = n_classes
        self.dropout   = dropout_p
        self.down_path = nn.ModuleList()
        self.down_path.append(ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0]))
        self.down_path.append(DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[0]))
        self.down_path.append(DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[0]))
        self.down_path.append(DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[0]))
        if(len(self.ft_chns) == 5):
            self.down_path.append(DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[0]))
    def forward(self, x):
        blocks=[]
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
        return blocks, x
    
class Decoder(nn.Module):
    def __init__(self, 
        in_chns,
        n_classes,
        ft_chns,
        dropout_p,
        trilinear):
        super().__init__()
        self.in_chns   = in_chns
        self.ft_chns   = ft_chns
        self.n_class   = n_classes
        self.dropout   = dropout_p
        self.trilinear = trilinear
        self.up_path = nn.ModuleList()
        if(len(self.ft_chns) == 5):
            self.up_path.append(UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], 
               dropout_p = self.dropout[1], trilinear=self.trilinear) )
        self.up_path.append(UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], 
               dropout_p = self.dropout[0], trilinear=self.trilinear) )
        self.up_path.append(UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], 
               dropout_p = self.dropout[0], trilinear=self.trilinear))
        self.up_path.append(UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], 
               dropout_p = self.dropout[0], trilinear=self.trilinear) )
        self.last = nn.Conv3d(self.ft_chns[0], self.n_class, kernel_size = 1)
    def forward(self, x, blocks):
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i -1])
        return self.last(x)
    
class UNet3d(nn.Module):
    def __init__(
        self, params
    ):
        super(UNet3d, self).__init__()
        lr = params['train']['lr']
        lr = params['train']['lr']
        dataset = params['train']['dataset']
        if dataset == 'fb' or dataset == 'feta':
            in_chns = params['network']['in_chns']
            n_classes = params['network']['n_classes_fb']
            ft_chns = params['network']['ft_chns_fb']
            self.pl_threshold = params['train']['pl_threshold_fb']
        if dataset == 'feta':
            in_chns = params['network']['in_chns']
            n_classes = params['network']['n_classes_feta']
            ft_chns = params['network']['ft_chns_feta']
            self.pl_threshold = params['train']['pl_threshold_feta']
        if dataset == 'mms':
            in_chns = params['network']['in_chns']
            n_classes = params['network']['n_classes_mms']
            ft_chns = params['network']['ft_chns_mms']
            self.pl_threshold = params['train']['pl_threshold_mms']
        dropout_p = params['network']['dropout_p']
        
        self.enc = Encoder(in_chns,n_classes,ft_chns,dropout_p)
        self.aux_dec1 = Decoder(in_chns,n_classes,ft_chns,dropout_p,trilinear=True)
        self.aux_dec2 = Decoder(in_chns,n_classes,ft_chns,dropout_p,trilinear=True)
        self.aux_dec3 = Decoder(in_chns,n_classes,ft_chns,dropout_p,trilinear=True)
        self.aux_dec4 = Decoder(in_chns,n_classes,ft_chns,dropout_p,trilinear=True)
        # setting the optimzer
        opt = 'adam'
        if opt == 'adam':
            self.enc_opt = torch.optim.Adam(self.enc.parameters(),lr=lr,betas=(0.9,0.999))
            self.aux_dec1_opt = torch.optim.Adam(self.aux_dec1.parameters(),lr=lr,betas=(0.5,0.999))
            self.aux_dec2_opt = torch.optim.Adam(self.aux_dec2.parameters(),lr=lr,betas=(0.5,0.999))
            self.aux_dec3_opt = torch.optim.Adam(self.aux_dec3.parameters(),lr=lr,betas=(0.5,0.999))
            self.aux_dec4_opt = torch.optim.Adam(self.aux_dec4.parameters(),lr=lr,betas=(0.5,0.999))
        elif opt == 'SGD':
            self.enc_opt = torch.optim.SGD(self.enc.parameters(),lr=lr,momentum=0.9)
            self.aux_dec1_opt = torch.optim.SGD(self.aux_dec1.parameters(),lr=lr,momentum=0)
            self.aux_dec2_opt = torch.optim.SGD(self.aux_dec2.parameters(),lr=lr,momentum=0)
            self.aux_dec3_opt = torch.optim.SGD(self.aux_dec3.parameters(),lr=lr,momentum=0)
            self.aux_dec4_opt = torch.optim.SGD(self.aux_dec4.parameters(),lr=lr,momentum=0)

        self.enc_opt_sch = get_scheduler(self.enc_opt)
        self.dec_1_opt_sch = get_scheduler(self.aux_dec1_opt)
        self.dec_2_opt_sch = get_scheduler(self.aux_dec2_opt)
        self.dec_3_opt_sch = get_scheduler(self.aux_dec3_opt)
        self.dec_4_opt_sch = get_scheduler(self.aux_dec4_opt)

    def initialize(self):
        init_weights(self.enc)
        init_weights(self.aux_dec1)
        init_weights(self.aux_dec2)
        init_weights(self.aux_dec3)
        init_weights(self.aux_dec4)
    def update_lr(self):
        self.enc_opt_sch.step()
        self.dec_1_opt_sch.step()
        self.dec_2_opt_sch.step()
        self.dec_3_opt_sch.step()
        self.dec_4_opt_sch.step()


    def forward_one_decoder(self, x):
        A_1 = rotate_single_with_label(x, 1)
        A_2 = rotate_single_with_label(x, 2)
        A_3 = rotate_single_with_label(x, 3)
        blocks1, latent_A1 = self.enc(A_1)
        blocks2, latent_A2 = self.enc(A_2)
        blocks3, latent_A3 = self.enc(A_3)
        blocks4, latent_A4 = self.enc(x)
        self.aux_seg_1 = self.aux_dec1(latent_A1, blocks1)
        self.aux_seg_2 = self.aux_dec1(latent_A2, blocks2)
        self.aux_seg_3 = self.aux_dec1(latent_A3, blocks3)
        self.aux_seg_4 = self.aux_dec1(latent_A4, blocks4)
        self.aux_seg_1 = rotate_single_with_label(self.aux_seg_1,3)
        self.aux_seg_2 = rotate_single_with_label(self.aux_seg_2,2)
        self.aux_seg_3 = rotate_single_with_label(self.aux_seg_3,1)
    def forward(self, x):
        A_1 = torch.rot90(x, 1, dims=(2, 3))
        A_2 = torch.rot90(x, 2, dims=(2, 3))
        A_3 = torch.rot90(x, 3, dims=(2, 3))
        blocks1, latent_A1 = self.enc(A_1)
        blocks2, latent_A2 = self.enc(A_2)
        blocks3, latent_A3 = self.enc(A_3)
        blocks4, latent_A4 = self.enc(x)
        self.aux_seg_1 = self.aux_dec1(latent_A1, blocks1).softmax(1)
        self.aux_seg_2 = self.aux_dec2(latent_A2, blocks2).softmax(1)
        self.aux_seg_3 = self.aux_dec3(latent_A3, blocks3).softmax(1)
        self.aux_seg_4 = self.aux_dec4(latent_A4, blocks4).softmax(1)
        self.aux_seg_1 = torch.rot90(self.aux_seg_1, 3, dims=(2, 3))
        self.aux_seg_2 = torch.rot90(self.aux_seg_2, 2, dims=(2, 3))
        self.aux_seg_3 = torch.rot90(self.aux_seg_3, 1, dims=(2, 3))
        return (self.aux_seg_1+self.aux_seg_2+self.aux_seg_3+self.aux_seg_4)/4.0
    
    def forward_source(self, x):
        blocks, latent_A = self.enc(x)
        self.aux_seg_1 = self.aux_dec1(latent_A, blocks).softmax(1)
        return self.aux_seg_1


    def save_nii(self,imagesb):
        self.forward(imagesb)
        pred_aux1 = self.aux_seg_1.cpu().detach().numpy()
        pred_aux2 = self.aux_seg_2.cpu().detach().numpy()
        pred_aux3 = self.aux_seg_3.cpu().detach().numpy()
        pred_aux4 = self.aux_seg_4.cpu().detach().numpy()
        self.four_predict_map = (pred_aux3+pred_aux4+pred_aux2+pred_aux1)/4.0
        self.four_predict_map[self.four_predict_map > self.pl_threshold] = 1
        self.four_predict_map[self.four_predict_map < 1] = 0
        B,C,D,W,H = self.four_predict_map.shape
        for j in range(B): 
            for i in range(C):
                self.four_predict_map[j,i,:,:,:] = get_largest_component(self.four_predict_map[j,i,:,:,:])


    def train_source(self,imagesa,labelsa):
        self.imgA = imagesa
        self.labA = labelsa
        self.forward(self.imgA)
        self.enc_opt.zero_grad()
        self.aux_dec1_opt.zero_grad()
        seg_loss_B = dice_loss(self.aux_seg_1,self.labA)
        seg_loss_B.backward()
        self.enc_opt.step()
        self.aux_dec1_opt.step()
        print(seg_loss_B.item())

    def trian_target(self,B):
        self.forward_44(B)
        self.enc_opt.zero_grad()
        self.aux_dec1_opt.zero_grad()
        self.aux_dec2_opt.zero_grad()
        self.aux_dec3_opt.zero_grad()
        self.aux_dec4_opt.zero_grad()
        device = B.device
        pseudo_lab = torch.from_numpy(self.four_predict_map.copy()).float().to(device) 
        size_b,size_c,size_d,size_w,size_h = pseudo_lab.shape 
        eara1 = self.aux_seg_1 * pseudo_lab
        eara2 = self.aux_seg_2 * pseudo_lab
        eara3 = self.aux_seg_3 * pseudo_lab
        eara4 = self.aux_seg_4 * pseudo_lab
        diceloss = dice_loss(eara4,pseudo_lab)+dice_loss(eara3,pseudo_lab)+dice_loss(eara2,pseudo_lab)+dice_loss(eara1,pseudo_lab)
        mean_map =  (self.aux_seg_2+self.aux_seg_4 +self.aux_seg_3+self.aux_seg_1) / 4.0
        mean_map_entropyloss = -(mean_map * torch.log2(mean_map + 1e-10)).sum() / (size_c*size_d*size_b*size_w*size_h)
        all_loss = diceloss + mean_map_entropyloss
        all_loss.backward()
        self.enc_opt.step()
        self.aux_dec4_opt.step()
        self.aux_dec3_opt.step()
        self.aux_dec2_opt.step()
        self.aux_dec1_opt.step()
        diceloss = diceloss.item()

        mean_map_entropyloss = mean_map_entropyloss.item()
        print('deiceloss:        ',diceloss)
        print('mean_entropyloss: ',mean_map_entropyloss)
        
        return diceloss,mean_map_entropyloss