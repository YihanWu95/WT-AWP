'''Modified from https://github.com/csdongxian/AWP/blob/main/AT_AWP/utils_awp.py 
   "Adversarial Weight Perturbation Helps Robust Generalization"  '''
'''WT-AWP for deep-robust models'''
import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F



def diff_in_weights(model, proxy,gamma):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k and not 'delta' in old_k:
            diff_w = new_w - old_w
            key = old_k+'_delta'
            diff_dict[key] = gamma * old_w.norm() / (diff_w.norm()+1e-5) * diff_w
    return diff_dict

def add_into_weights(model, diff, coeff=1.0,restore = False):
    names_in_diff = diff.keys()
    for name, param in model.named_parameters():
        if name in names_in_diff:
            if restore:
                param.zero_()
            else:    
                param.add_(coeff * diff[name])


class AdvWeightPerturb(object):
    def __init__(self, model, proxy, proxy_optim, gamma, step, attack_method='PGD',model_name = ""):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma
        self.step = step
        self.lr = gamma/step
        self.attack_method = attack_method
        self.model_name = model_name

    def calc_awp(self, x,adj,train_mask,y):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        
        for i in range(self.step):
            self.proxy_optim.zero_grad()
            loss = - F.nll_loss(self.proxy(x,adj)[train_mask], y[train_mask])
            loss.backward()
            self.proxy_optim.step()


        diff = diff_in_weights(self.model, self.proxy,self.gamma)
        return diff

    def perturb(self, diff, epoch=0):
        add_into_weights(self.model, diff, coeff=1)

    def restore(self, diff):
        add_into_weights(self.model, diff, restore = True)
        