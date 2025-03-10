import torch
import torch.nn.functional as F
import torch.nn as nn


class losses_computer():
    def __init__(self, opt):
        self.opt = opt
        if not opt.no_labelmix:
            self.labelmix_function = torch.nn.MSELoss()

    def loss(self, input, label, for_real):
        #--- balancing classes ---
        weight_map = get_class_balancing(self.opt, input, label)
        #--- n+1 loss ---
        target = get_n1_target(self.opt, input, label, for_real)
        loss = F.cross_entropy(input, target, reduction='none')
        if for_real:
            loss = torch.mean(loss * weight_map[:, 0, :, :])
        else:
            loss = torch.mean(loss)
        return loss

    def loss_labelmix(self, mask, output_D_mixed, output_D_fake, output_D_real):
        mixed_D_output = mask*output_D_real+(1-mask)*output_D_fake
        return self.labelmix_function(mixed_D_output, output_D_mixed)


def get_class_balancing(opt, input, label):
    if not opt.no_balancing_inloss:
        class_occurence = torch.sum(label, dim=(0, 2, 3))
        if opt.contain_dontcare_label:
            class_occurence[0] = 0
        num_of_classes = (class_occurence > 0).sum()
        coefficients = torch.reciprocal(class_occurence) * torch.numel(label) / (num_of_classes * label.shape[1])
        integers = torch.argmax(label, dim=1, keepdim=True)
        if opt.contain_dontcare_label:
            coefficients[0] = 0
        weight_map = coefficients[integers]
    else:
        weight_map = torch.ones_like(input[:, :, :, :])
    return weight_map


def generate_labelmix(label, fake_image, real_image):
    target_map = torch.argmax(label, dim = 1, keepdim = True)
    all_classes = torch.unique(target_map)
    for c in all_classes:
        target_map[target_map == c] = torch.randint(0,2,(1,))
    target_map = target_map.float()
    mixed_image = target_map*real_image+(1-target_map)*fake_image
    return mixed_image, target_map


def get_n1_target(opt, input, label, target_is_real):
    targets = get_target_tensor(opt, input, target_is_real)
    num_of_classes = label.shape[1]
    integers = torch.argmax(label, dim=1)
    targets = targets[:, 0, :, :] * num_of_classes
    integers += targets.long()
    integers = torch.clamp(integers, min=num_of_classes-1) - num_of_classes + 1
    return integers


def get_target_tensor(opt, input, target_is_real):
    if opt.gpu_id != "-1":
        if target_is_real:
            return torch.cuda.FloatTensor(1).fill_(1.0).requires_grad_(False).expand_as(input)
        else:
            return torch.cuda.FloatTensor(1).fill_(0.0).requires_grad_(False).expand_as(input)
    else:
        if target_is_real:
            return torch.FloatTensor(1).fill_(1.0).requires_grad_(False).expand_as(input)
        else:
            return torch.FloatTensor(1).fill_(0.0).requires_grad_(False).expand_as(input)