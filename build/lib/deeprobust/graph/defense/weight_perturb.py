import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
EPS = 1e-5


def diff_in_weights(model, proxy, attack_method='FSGM'):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
#         print(old_w.requires_grad)

        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k and not 'delta' in old_k:
            diff_w = new_w - old_w
            key = old_k+'_delta'
#             print(key)
            if attack_method == 'FSGM':
#                 diff_w = torch.sign(diff_w)
                diff_w = diff_w
#                 print(diff_w)
            elif attack_method == 'random':
                diff_w = torch.sign(torch.randn_like(diff_w))
            diff_dict[key] = torch.clamp(old_w.norm() / (diff_w.norm() + EPS) * diff_w,max = 10, min = -10)

#             diff_dict[key] = torch.sign(diff_w)#old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


# def add_into_weights(model, diff, coeff=1.0):
#     names_in_diff = diff.keys()
#     with torch.no_grad():
#         for name, param in model.named_parameters():
#             if name in names_in_diff:
#                 param.add_(coeff * diff[name])

def add_into_weights(model, diff, coeff=1.0,restore = False):
    names_in_diff = diff.keys()
#     with torch.no_grad():
#     print(names_in_diff)
    for name, param in model.named_parameters():
#         print(name)
        if name in names_in_diff:
#             print(param.requires_grad)
#                 print(name)
            if restore:
                param.zero_()
#                 print(param)
            else:    
                param.add_(coeff * diff[name])
#             print(param)
#                     print(param)

class AdvWeightPerturb(object):
    def __init__(self, model, proxy, proxy_optim, gamma, step, attack_method='FSGM'):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma
        self.step = step
        self.lr = gamma/step
        self.attack_method = attack_method
#         self.model_type = self

    def calc_awp(self, x,adj,train_mask,y):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
#         self.proxy_optim.zero_grad()
        
#         diff_dict = OrderedDict()
        
#         for i in range(self.step):
#             for name, param in self.proxy.named_parameters():
#                 if 'weight' in name and not 'delta' in name:
#                     print(param)
#                     name_new = name+'_delta'
#                     grad = torch.sign(torch.autograd.grad(loss, param)[0].data)
# #                     param.add_(self.lr*grad)
#                     param = param + (self.lr*grad)
#                     print(param)
        if self.attack_method == 'FSGM':
#             print(1)
            for i in range(self.step):
                self.proxy_optim.zero_grad()
    #             if self.model_type == 'GCN':
    #                 loss = -F.nll_loss(self.proxy()[data.train_mask], data.y[data.train_mask]) 
    #             elif self.model_type == 'MultiGCN':
    #                 out = proxy(data.x, data.adj_t)
    #                 loss = -F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    #             loss = - F.nll_loss(self.proxy(data)[data.train_mask], data.y[data.train_mask])
                loss = - F.nll_loss(self.proxy(x,adj)[train_mask], y[train_mask])
                loss.backward()
                self.proxy_optim.step()

        # the adversary weight perturb
#         print(self.lr)
        diff = diff_in_weights(self.model, self.proxy, self.attack_method)
#         print(diff)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, restore = True)