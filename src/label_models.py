from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from copy import deepcopy

from snorkel.labeling.model import LabelModel as SnorkelLM

import optuna
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import roc_auc_score, f1_score

optuna.logging.disable_default_handler()


def discard(L_tr, xs_tr, L_val, xs_val, anchors, ratio=0.2):
    if ratio == 0:
        return L_tr, L_val

    xs_tr_val = np.vstack((xs_tr, xs_val))
    L_tr_val = np.vstack((L_tr, L_val))
    L_tr_val_discarded = np.zeros_like(L_tr_val)
    num_lfs = L_tr.shape[1]
    n_splits = int(1. / ratio)
    for j in range(num_lfs):
        anchor = anchors[j]
        labels = L_tr_val[:, j]

        dists_tr_val = cosine_distances(anchor.reshape(1, -1), xs_tr_val)[0]
        scaler = KBinsDiscretizer(n_bins=n_splits, encode='ordinal', strategy='quantile')
        xs_tr_val_discrete = scaler.fit_transform(dists_tr_val.reshape(-1, 1)).reshape(-1)

        discard_mask = (xs_tr_val_discrete == (n_splits - 1))
        labels[discard_mask] = 0

        L_tr_val_discarded[:, j] = labels

    L_tr_discarded = L_tr_val_discarded[:len(L_tr)]
    L_val_discarded = L_tr_val_discarded[len(L_tr):]

    return L_tr_discarded, L_val_discarded



def merge_kwargs(kwargs_0, kwargs_1):
    for key, value in kwargs_1.items():
        kwargs_0[key] = value

    return kwargs_0


def get_label_model(method, **kwargs):
    if method == 'mv':
        mv_kwargs = defaultdict(dict)

        if kwargs['discard'] is None:
            pass
        elif kwargs['discard'] == 'grid':
            mv_kwargs['search_space']['discard'] = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
            mv_kwargs['n_trials'] = 256
        else:
            raise NotImplementedError

        kwargs = merge_kwargs(kwargs, mv_kwargs)

        return MajorityVote(**kwargs)

    elif method == 'snorkel':
        snorkel_kwargs = {
            'search_space': {
                # 'optimizer': ['sgd'],
                'lr': np.logspace(-4, -1, num=4, base=10),
                'l2': np.logspace(-4, -1, num=4, base=10),
                'n_epochs': [5, 10, 50, 100],
            },
            'n_trials': 512
        }

        if kwargs['discard'] is None:
            pass
        elif kwargs['discard'] == 'grid':
            snorkel_kwargs['search_space']['discard'] = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        else:
            raise NotImplementedError

        kwargs = merge_kwargs(kwargs, snorkel_kwargs)

        return Snorkel(**kwargs)

    elif method == 'unipolar':
        unipolar_kwargs = {
            'search_space':{
                'init_tpr_fpr_ratio': [2.],
                'class_prior_ratio': [1.], 
                'learn_prior': [False],
                'epochs': [50, 100], 
                'lr': np.logspace(-4, -1, num=4, base=10), 
                'weight_decay': [0., 1e-3, 1e-2, 1e-1],
                'batch_size': [256],
                'tolerance': [3]
            },
            'n_trials': 200
        }
        kwargs = merge_kwargs(kwargs, unipolar_kwargs)

        return Unipolar(**kwargs)
    else:
        raise NotImplementedError


class LabelModel(object):
    def fit(self, L_tr, L_val, ys_val, xs_tr=None, xs_val=None):
        raise NotImplementedError

    def predict_proba(self, L):
        raise NotImplementedError

    def predict(self, L):
        ys_pred = self.predict_proba(L)
        # breaking ties by random selection
        ys_pred = np.array([np.random.choice(np.where(y==np.max(y))[0]) for y in ys_pred])
        ys_pred[ys_pred==0] = -1
        return ys_pred


def to_snorkel_L(L):
    snorkel_L = np.copy(L)
    snorkel_L[L==1] = 1
    snorkel_L[L==0] = -1
    snorkel_L[L==-1] = 0

    return snorkel_L


def snorkel_to_normal_L(L):
    normal_L = np.copy(L)
    normal_L[L==1] = 1
    normal_L[L==-1] = 0
    normal_L[L==0] = -1

    return normal_L



def filter_abstain(L, ys=None):
    non_abstain = (L!=0).any(axis=1)
    L = L[non_abstain]
    if ys is not None:
        ys = ys[non_abstain]
        return L, ys, non_abstain
    else:
        return L, non_abstain


class Snorkel(LabelModel):
    def __init__(self, **kwargs):
        self.search_space = kwargs['search_space']
        self.n_trials = kwargs.get('n_trials', 100) 
        self.best_params = None
        self.model = None
        self.num_lfs = kwargs['num_lfs']
        self.anchors = kwargs['anchors']
        self.kwargs = kwargs


    def _to_onehot(self, ys):
        ys = np.array(ys)
        ys[ys==-1] = 0
        ys_onehot = np.zeros((len(ys), 2))
        ys_onehot[range(len(ys_onehot)), ys] = 1

        return ys_onehot


    def fit(self, L_tr, L_val, ys_val, xs_tr=None, xs_val=None, scoring='logloss'):
        seed = np.random.randint(1e6)
        search_space = self.search_space
        anchors = self.anchors

        if self.kwargs['discard'] is None:
            L_tr, L_val = to_snorkel_L(L_tr), to_snorkel_L(L_val)
        elif self.kwargs['discard'] == 'grid':
            assert xs_tr is not None and xs_val is not None
            L_dict = defaultdict(dict)
            for ratio in search_space['discard']:
                L_tr_dis, L_val_dis = discard(L_tr, xs_tr, L_val, xs_val, anchors, ratio)
                L_tr_dis, filter_mask_tr = filter_abstain(L_tr_dis)
                L_val_dis, ys_val_dis, _ = filter_abstain(L_val_dis, ys_val)
                L_dict[ratio]['L_tr'], L_dict[ratio]['L_val'] = to_snorkel_L(L_tr_dis), to_snorkel_L(L_val_dis)
                L_dict[ratio]['ys_val'] = ys_val_dis
                L_dict[ratio]['filter_mask_tr'] = filter_mask_tr
        elif self.kwargs['discard'] == 'grid-tr':
            raise NotImplementedError
        else:
            raise NotImplementedError


        def objective(trial):
            suggestions = {key: trial.suggest_categorical(key, search_space[key]) for key in search_space}
            
            if self.kwargs['discard'] is None:
                nonlocal L_tr, L_val, ys_val
            elif self.kwargs['discard'] == 'grid-tr':
                raise NotImplementedError
            elif self.kwargs['discard'] == 'grid':
                L_tr, L_val = L_dict[suggestions['discard']]['L_tr'], L_dict[suggestions['discard']]['L_val']
                ys_val = L_dict[suggestions['discard']]['ys_val']
                suggestions.pop('discard')
            else:
                raise NotImplementedError
                

            model = SnorkelLM(cardinality=2, verbose=True)

            model.fit(L_train=L_tr, class_balance=self.kwargs['class_balance'], **suggestions, seed=seed)            
            # model.fit(L_train=L_tr, Y_dev=ys_val, **suggestions, seed=seed)            

            # logloss as validation loss
            if scoring == 'logloss':
                ys_pred_val = model.predict_proba(L_val)
                ys_val_onehot = self._to_onehot(ys_val)
                val_loss = -(ys_val_onehot * np.log(np.clip(ys_pred_val, 1e-6, 1.))).sum(axis=1).mean()
            elif scoring == 'f1':
                ys_pred = model.predict(L_val)
                val_loss = -f1_score(ys_val, snorkel_to_normal_L(ys_pred))

            return val_loss

        # search for best hyperparameter
        
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)

        self.best_params = study.best_params

        # discarding
        if self.kwargs['discard'] is None:
            filter_mask_tr = np.array([True] * len(L_tr))
        elif self.kwargs['discard'] == 'grid':
            L_tr = L_dict[self.best_params['discard']]['L_tr']
            ys_val = L_dict[self.best_params['discard']]['ys_val']
            filter_mask_tr = L_dict[self.best_params['discard']]['filter_mask_tr']
            self.best_params.pop('discard')
        elif self.kwargs['discard'] == 'grid-tr':
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.model = SnorkelLM(cardinality=2, verbose=True)

        self.model.fit(L_train=L_tr, class_balance=self.kwargs['class_balance'], **self.best_params, seed=seed)            
        # self.model.fit(L_train=L_tr, Y_dev=ys_val, **self.best_params, seed=seed)            
        L_tr = snorkel_to_normal_L(L_tr)

        np.random.seed(seed)

        return L_tr, filter_mask_tr

        
    def predict_proba(self, L):
        L = to_snorkel_L(L)
        return self.model.predict_proba(L)


class MajorityVote(LabelModel):
    def __init__(self, **kwargs):
        self.search_space = kwargs.get('search_space', None)
        self.n_trials = kwargs.get('n_trials', 100) 
        self.best_params = None
        self.num_lfs = kwargs['num_lfs']
        self.anchors = kwargs['anchors']
        self.kwargs = kwargs


    def _to_onehot(self, ys):
        ys = np.array(ys)
        ys[ys==-1] = 0
        ys_onehot = np.zeros((len(ys), 2))
        ys_onehot[range(len(ys_onehot)), ys] = 1

        return ys_onehot


    def fit(self, L_tr, L_val, ys_val, xs_tr, xs_val):
        search_space = self.search_space
        anchors = self.anchors

        if search_space is None:
            return L_tr, np.array([True] * len(L_tr))

        if self.kwargs['discard'] is None:
            pass
        elif self.kwargs['discard'] == 'grid':
            assert xs_tr is not None and xs_val is not None
            L_dict = defaultdict(dict)
            for ratio in search_space['discard']:
                L_tr_dis, L_val_dis = discard(L_tr, xs_tr, L_val, xs_val, anchors, ratio)
                L_tr_dis, filter_mask_tr = filter_abstain(L_tr_dis)
                L_val_dis, ys_val_dis, _ = filter_abstain(L_val_dis, ys_val)
                L_dict[ratio]['L_tr'], L_dict[ratio]['L_val'] = L_tr_dis, L_val_dis
                L_dict[ratio]['ys_val'] = ys_val_dis
                L_dict[ratio]['filter_mask_tr'] = filter_mask_tr
        else:
            raise NotImplementedError


        def objective(trial):
            suggestions = {key: trial.suggest_categorical(key, search_space[key]) for key in search_space}
            
            if self.kwargs['discard'] is None:
                nonlocal L_tr, L_val, ys_val
            elif self.kwargs['discard'] == 'grid':
                L_tr, L_val = L_dict[suggestions['discard']]['L_tr'], L_dict[suggestions['discard']]['L_val']
                ys_val = L_dict[suggestions['discard']]['ys_val']
            else:
                raise NotImplementedError

            # logloss as validation loss
            ys_pred_val = self.predict_proba(L_val)
            ys_val_onehot = self._to_onehot(ys_val)
            val_loss = -(ys_val_onehot * np.log(np.clip(ys_pred_val, 1e-6, 1.))).sum(axis=1).mean()

            return val_loss

        # search for best hyperparameter
        
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)

        self.best_params = study.best_params

        # discarding
        if self.kwargs['discard'] is None:
            filter_mask_tr = np.array([True] * len(L_tr))
        elif self.kwargs['discard'] == 'grid':
            L_tr = L_dict[self.best_params['discard']]['L_tr']
            filter_mask_tr = L_dict[self.best_params['discard']]['filter_mask_tr']
        else:
            raise NotImplementedError

        return L_tr, filter_mask_tr

    def predict_proba(self, L):
        y_preds = np.zeros((L.shape[0], 2))
        num_non_abstain = (L != 0).sum(axis=1)
        non_abstain = num_non_abstain > 0
        y_preds[non_abstain, 0] = (L[non_abstain] == -1).sum(axis=1) / num_non_abstain[non_abstain]
        y_preds[non_abstain, 1] = (L[non_abstain] == 1).sum(axis=1) / num_non_abstain[non_abstain]
        y_preds[~non_abstain, :] = 0.5

        return y_preds


#### unused modules below ####

def discard_tr(L_tr, xs_tr, anchors, ratio=0.2):
    if ratio == 0:
        return L_tr
    
    L_tr_discarded = np.zeros_like(L_tr)
    num_lfs = L_tr.shape[1]
    n_splits = int(1. / ratio)

    for j in range(num_lfs):
        anchor = anchors[j]
        labels = L_tr[:, j]

        dists_tr = cosine_distances(anchor.reshape(1, -1), xs_tr)[0]
        # dists_tr = euclidean_distances(anchor.reshape(1, -1), xs_tr)[0]
        scaler = KBinsDiscretizer(n_bins=n_splits, encode='ordinal', strategy='quantile')
        # scaler = KBinsDiscretizer(n_bins=n_splits, encode='ordinal', strategy='kmeans')
        xs_tr_discrete = scaler.fit_transform(dists_tr.reshape(-1, 1)).reshape(-1)

        discard_mask = (xs_tr_discrete == (n_splits - 1))
        labels[discard_mask] = 0

        L_tr_discarded[:, j] = labels

    return L_tr_discarded

class Unipolar(LabelModel):
    def __init__(self, **kwargs):
        self.search_space = kwargs['search_space']
        self.n_trials = kwargs.get('n_trials') 
        self.best_params = None
        self.model = None
        self.kwargs = kwargs
        self.num_lfs = kwargs['num_lfs']
        self.lf_labels = kwargs['lf_labels']

    def _to_onehot(self, ys):
        ys = np.array(ys)
        ys[ys==-1] = 0
        ys_onehot = np.zeros((len(ys), 2))
        ys_onehot[range(len(ys_onehot)), ys] = 1

        return ys_onehot

    def fit(self, L_tr, L_val, ys_val, xs_tr=None, xs_val=None):
        seed = np.random.randint(1e6)
        search_space = self.search_space
        num_lfs = self.num_lfs
        lf_labels = self.lf_labels

        def objective(trial):
            suggestions = {key: trial.suggest_categorical(key, search_space[key]) for key in search_space}
            model = UnipolarLM(num_lfs=num_lfs, lf_labels=lf_labels, **suggestions)
            model.fit(L_tr=L_tr, L_val=L_val, ys_val=ys_val, seed=seed)
            ys_pred_val = model.predict_proba(L_val)
            ys_val_onehot = self._to_onehot(ys_val)
            val_loss = -(ys_val_onehot * np.log(np.clip(ys_pred_val, 1e-6, 1.))).sum(axis=1).mean()

            return val_loss

        # search for best hyperparameter
        
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)

        self.best_params = study.best_params

        self.model = UnipolarLM(num_lfs=num_lfs, lf_labels=lf_labels, **self.best_params)
        self.model.fit(L_tr=L_tr, L_val=L_val, ys_val=ys_val, seed=seed)
        np.random.seed(seed)


    def predict_proba(self, L):
        return self.model.predict_proba(L)


class LMDataset(Dataset):
    def __init__(self, L, ys=None):
        super().__init__()
        self.L = L
        self.ys = ys

    def __getitem__(self, index):
        if self.ys is None:
            return self.L[index]
        else:
            return self.L[index], self.ys[index]

    def __len__(self):
        return self.L.shape[0]


class UnipolarLM(nn.Module):
    def __init__(self, num_lfs, lf_labels, init_tpr=0.05, init_tpr_fpr_ratio=2., class_prior_ratio=1., learn_prior=False,
                 epochs=50, lr=0.01, weight_decay=0., batch_size=256, tolerance=3, **kwargs):
        super().__init__()
        
        self.num_lfs = num_lfs
        self.lf_labels = torch.tensor(lf_labels)

        self.init_tpr_fpr_ratio = init_tpr_fpr_ratio
        # init_fpr = init_tpr / init_tpr_fpr_ratio

        # init_tpr_logit = torch.logit(torch.tensor(init_tpr))
        # init_fpr_logit = torch.logit(torch.tensor(init_fpr))

        # self._init_params(init_tpr_logit, init_fpr_logit) # logits of conditional prob P(y_hat!=0 | y)

        self.learn_prior = learn_prior
        class_priors = [class_prior_ratio / (1. + class_prior_ratio), 1. / (1. + class_prior_ratio)]
        self.class_priors = nn.Parameter(torch.tensor(class_priors), requires_grad=learn_prior)
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.tolerance = tolerance

    
    def _init_params(self, L_tr):
        """Assume unipolar lfs for binary classification task
        """
        p_rates = (L_tr != 0).mean(axis=0)
        init_tprs = p_rates * (self.init_tpr_fpr_ratio / (self.init_tpr_fpr_ratio + 1.))
        init_fprs = p_rates * (1. / (self.init_tpr_fpr_ratio + 1.))

        init_tpr_logits = torch.logit(torch.tensor(init_tprs)).float()
        init_fpr_logits = torch.logit(torch.tensor(init_fprs)).float()
        # init_tpr_logits = torch.full([self.num_lfs], init_tpr_logit)
        # init_fpr_logits = torch.full([self.num_lfs], init_fpr_logit)
        q_nz_pos = np.where(self.lf_labels==1, init_tpr_logits, init_fpr_logits)
        q_nz_neg = np.where(self.lf_labels==-1, init_tpr_logits, init_fpr_logits)

        self.q_nz_pos = nn.Parameter(torch.tensor(q_nz_pos), requires_grad=True)
        self.q_nz_neg = nn.Parameter(torch.tensor(q_nz_neg), requires_grad=True)
        

    def forward(self, L):
        L_nz = (L != 0).float()

        q_nz_pos = self.q_nz_pos
        q_nz_neg = self.q_nz_neg
        p_nz_pos = torch.clamp(torch.sigmoid(q_nz_pos), 1e-6, 1-1e-6)
        p_nz_neg = torch.clamp(torch.sigmoid(q_nz_neg), 1e-6, 1-1e-6)

        cond_ll_pos = (L_nz * q_nz_pos).sum(dim=1, keepdim=True) + torch.log(1. - p_nz_pos).sum()
        cond_ll_neg = (L_nz * q_nz_neg).sum(dim=1, keepdim=True) + torch.log(1. - p_nz_neg).sum()

        cond_ll = torch.cat((cond_ll_neg, cond_ll_pos), dim=1)
        cond_ll = cond_ll + self.class_priors
        log_likelihood = torch.logsumexp(cond_ll, dim=1)

        return log_likelihood, cond_ll


    def _to_onehot(self, ys):
        ys[ys==-1] = 0
        ys_onehot = torch.zeros((len(ys), 2))
        ys_onehot[range(len(ys_onehot)), ys] = 1.

        return ys_onehot


    def fit(self, L_tr, L_val, ys_val, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

        # init params
        self._init_params(L_tr)

        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # optimizer = torch.optim.Adamax(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        dataset_tr = LMDataset(L_tr)
        dataset_val = LMDataset(L_val, ys_val)
        data_loader_tr = DataLoader(dataset_tr, batch_size=self.batch_size, shuffle=True) 
        data_loader_val = DataLoader(dataset_val, batch_size=L_val.shape[0], shuffle=False) 
    
        pre_val_loss = np.inf
        best_params = None
        for epoch in range(self.epochs):
            running_loss = 0.
            for data_tr in data_loader_tr: 
                optimizer.zero_grad()
                log_likelihood, _ = self(data_tr)
                loss = -log_likelihood.mean()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(data_loader_tr)                

            val_loss = 0.
            for data_val, ys_val in data_loader_val:
                ys_pred = self._get_posterior(data_val)
                ys_val_onehot = self._to_onehot(ys_val)
                val_loss += F.binary_cross_entropy(ys_pred, ys_val_onehot).item()
            val_loss /= len(data_loader_val)

            if val_loss < pre_val_loss:
                pre_val_loss = val_loss
                best_params = deepcopy(self.state_dict())
                tolerance = self.tolerance
            else:
                tolerance -= 1
            
            if tolerance <= 0:
                self.load_state_dict(best_params)
                break


    def _get_posterior(self, L):
        log_likelihood,  cond_ll = self(L)
        ys_pred = torch.exp(cond_ll - log_likelihood.view(-1, 1))
        
        return ys_pred


    def predict_proba(self, L):
        L = torch.tensor(L)
        return self._get_posterior(L).detach().numpy()