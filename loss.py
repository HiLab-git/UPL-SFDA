#loss function for UPL-SFDA
import torch
from torch import nn
import torch.nn.functional as F

def dice_weight_loss(predict,target,weight):
    target = target.float()*weight
    predict = predict*weight
    smooth = 1e-4
    intersect = torch.sum(predict*target)
    dice = (2 * intersect + smooth)/(torch.sum(target)+torch.sum(predict*predict)+smooth)
    loss = 1.0 - dice
    return loss
def dice_loss(predict,target):
    target = target.float()
    smooth = 1e-4
    intersect = torch.sum(predict*target)
    dice = (2 * intersect + smooth)/(torch.sum(target)+torch.sum(predict*predict)+smooth)
    loss = 1.0 - dice
    return loss
class diceLoss_weight(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        self.n_classes = n_classes
    def one_hot_encode(self,input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            tmp = (input_tensor==i) * torch.ones_like(input_tensor)
            tensor_list.append(tmp)
        output_tensor = torch.cat(tensor_list,dim=1)
        return output_tensor.float()
    
    def forward(self,inputs,target,weight):
        x_shape = list(target.shape)
        if(len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            target = torch.transpose(target, 1, 2)
            target = torch.reshape(target, new_shape)
        target = self.one_hot_encode(target)
        
        assert inputs.shape == target.shape,(target.shape,inputs.shape)
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            diceloss = dice_weight_loss(inputs[:,i,:,:], target[:,i,:,:],weight)
            class_wise_dice.append(diceloss)
            loss += diceloss
        return loss/self.n_classes

class DiceLoss(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        self.n_classes = n_classes
        
    def one_hot_encode(self,input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            tmp = (input_tensor==i) * torch.ones_like(input_tensor)
            tensor_list.append(tmp)
        output_tensor = torch.cat(tensor_list,dim=1)
        return output_tensor.float()
    
    def forward(self,inputs,target,one_hot):
        x_shape = list(target.shape)
        if(len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            target = torch.transpose(target, 1, 2)
            target = torch.reshape(target, new_shape)

        if one_hot:
            target = self.one_hot_encode(target)
        assert inputs.shape == target.shape,'size must match'
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            diceloss = dice_loss(inputs[:,i,:,:], target[:,i,:,:])
            class_wise_dice.append(diceloss)
            loss += diceloss
        return loss/self.n_classes

class Ce_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,input,target):
        inputs = F.softmax(input,dim=1)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        loss = 0
        for i in range(0,input.shape[0]):
            loss += self.ce_loss(input[i].unsqueeze(0),target)
        return loss

class DiceLoss_n(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        self.n_classes = n_classes
    
    def forward(self,input,target,weight=None,softmax=True):
        if softmax:
            inputs = F.softmax(input,dim=1)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.shape == target.shape,'size must match'
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            diceloss = dice_loss(inputs[:,i], target[:,i])
            class_wise_dice.append(diceloss)
            loss += diceloss * weight[i]
        return loss/self.n_classes

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.eps = 1e-4
        self.num_classes = num_classes

    def forward(self, predict, target):
        weight = []
        for c in range(self.num_classes):
            weight_c = torch.sum(target == c).float()
            weight.append(weight_c)
        weight = torch.tensor(weight).to(target.device)
        weight = 1 - weight / (torch.sum(weight))
        weight[0] = 0.0
        target = target.argmax(axis=1)
        wce_loss = F.cross_entropy(predict, target.long(), weight)
        return wce_loss

class DiceLoss_weight(nn.Module):
    def __init__(self,num_classes,alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.diceloss = diceLoss_weight(self.num_classes)
    def forward(self,predict,label,weight):
        x_shape = list(label.shape)
        if(len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(label, 1, 2)
            label = torch.reshape(x, new_shape)
        loss = self.diceloss(predict,label,weight) 
        return loss
class DiceCeLoss(nn.Module):
     #predict : output of model (i.e. no softmax)[N,C,*]
     #target : gt of img [N,1,*]
    def __init__(self,num_classes,alpha=1.0):
        '''
        calculate loss:
            celoss + alpha*celoss
            alpha : default is 1
        '''
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.diceloss = DiceLoss(self.num_classes)
        self.celoss = WeightedCrossEntropyLoss(self.num_classes)
        
    def forward(self,predict,label,one_hot):
        
        x_shape = list(label.shape)
        if(len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(label, 1, 2)
            label = torch.reshape(x, new_shape)
        celoss = self.celoss.to(label.device)
        diceloss = self.diceloss(predict,label,one_hot)
        celoss = self.celoss(predict,label)
        loss = diceloss + celoss
        return loss
        