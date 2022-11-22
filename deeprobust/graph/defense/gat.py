"""
Extended from https://github.com/rusty1s/pytorch_geometric/tree/master/benchmark/citation
"""
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from .gat_conv import GATConv

from torch_geometric.nn import GATConv as GATConv2

from .weight_perturb_pyg import AdvWeightPerturb



class GAT(nn.Module):
    """ 2 Layer Graph Attention Network based on pytorch geometric.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    heads: int
        number of attention heads
    output_heads: int
        number of attention output heads
    dropout : float
        dropout rate for GAT
    lr : float
        learning rate for GAT
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN.
        When `with_relu` is True, `weight_decay` will be set to 0.
    with_bias: bool
        whether to include bias term in GAT weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
	We can first load dataset and then train GAT.

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GAT
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> gat = GAT(nfeat=features.shape[1],
              nhid=8, heads=8,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    >>> gat = gat.to('cpu')
    >>> pyg_data = Dpr2Pyg(data) # convert deeprobust dataset to pyg dataset
    >>> gat.fit(pyg_data, patience=100, verbose=True) # train with earlystopping
    """

    def __init__(self, nfeat, nhid, nclass, heads=8, output_heads=1, dropout=0.5, lr=0.01,
            weight_decay=5e-4, with_bias=True, device=None):

        super(GAT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        self.conv1 = GATConv(
            nfeat,
            nhid,
            heads=heads,
            dropout=dropout,
            bias=with_bias)

        self.conv2 = GATConv2(
            nhid * heads,
            nclass,
            heads=output_heads,
            concat=False,
            dropout=dropout,
            bias=with_bias)

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of GAT.
        """
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def fit(self, pyg_data, train_iters=200, initialize=True, verbose=False, patience=100, seed = 0,**kwargs):
        """Train the GAT model, when idx_val is not None, pick the best model
        according to the validation loss.

        Parameters
        ----------
        pyg_data :
            pytorch geometric dataset object
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """

        torch.manual_seed(seed)
        if initialize:
            self.initialize()

        self.data = pyg_data[0].to(self.device)
        # By default, it is trained with early stopping on validation
        self.train_with_early_stopping(train_iters, patience, verbose)
        
    def fit_mp(self, pyg_data, proxy, train_iters=200, initialize=True, verbose=False, patience=500, awp_weight = 1, awp_gamma = 1, awp_step = 5, awp_method = 'FSGM', idx_test = None, seed = 0, gammadecay = [20],gradient = False,**kwargs):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        normalize : bool
            whether to normalize the input adjacency matrix.
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """
        torch.manual_seed(seed)
        self.data = pyg_data[0].to(self.device)
        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        proxy_opt = torch.optim.SGD(proxy.parameters(), lr=1)
        awp_adversary = AdvWeightPerturb(model=self, proxy=proxy, proxy_optim=proxy_opt, gamma=awp_gamma,step=awp_step,attack_method = awp_method)
        if initialize:
            self.initialize()

        if verbose:
            print('=== training gcn model ===')
#         optimizer = optim.Adam([
#     dict(params=self.gc1.parameters(), weight_decay=5e-4),
#     dict(params=self.gc2.parameters(), weight_decay=0)], lr=self.lr)
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            
#             output = self.forward(self.data)

#             loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            output_nat = self.forward(self.data)
            loss_nat = F.nll_loss(output_nat[train_mask], labels[train_mask])
            awp = awp_adversary.calc_awp(data=self.data, train_mask = train_mask,y=labels)
            awp_adversary.perturb(awp,i)
            output_adv = self.forward(self.data)
            loss_adv = F.nll_loss(output_adv[train_mask], labels[train_mask])
            loss = (1-awp_weight)*loss_nat + awp_weight*loss_adv
            
            loss.backward()
            optimizer.step()
#             print(self.gc1.delta)
            awp_adversary.restore(awp)
#             print(self.gc1.delta)
            if verbose and i % 100 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss.item()))
                self.test(idx_test)

            self.eval()
            output = self.forward(self.data)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])
            acc_val = utils.accuracy(output[val_mask], labels[val_mask])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)


    def train_with_early_stopping(self, train_iters, patience, verbose):
        """early stopping based on the validation loss
        """
        if verbose:
            print('=== training GAT model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.data)

            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.data)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def test(self,test_idx=None):
        """Evaluate GAT performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        test_mask = self.data.test_mask
        labels = self.data.y
        output = self.forward(self.data)
        # output = self.output
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = utils.accuracy(output[test_mask], labels[test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def predict(self):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GAT
        """

        self.eval()
        return self.forward(self.data)



if __name__ == "__main__":
    from deeprobust.graph.data import Dataset, Dpr2Pyg
    # from deeprobust.graph.defense import GAT
    data = Dataset(root='/tmp/', name='cora')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    gat = GAT(nfeat=features.shape[1],
          nhid=8, heads=8,
          nclass=labels.max().item() + 1,
          dropout=0.5, device='cpu')
    gat = gat.to('cpu')
    pyg_data = Dpr2Pyg(data)
    gat.fit(pyg_data, verbose=True) # train with earlystopping
    gat.test()
    print(gat.predict())

