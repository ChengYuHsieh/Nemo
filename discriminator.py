import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from  torch.utils.data import Dataset, DataLoader
import optuna
from utils import *
import pdb


def get_discriminator(model_type, prob_labels, params=None, seed=None):
    if model_type == 'logistic':
        return LogReg(prob_labels, params, seed)
    elif model_type == 'torch':
        return LogRegTorch(prob_labels, params, seed)
    elif model_type == 'ssl':
        return LogRegEM(params, seed)
    elif model_type == 'svm':
        return SVM(params)
    else:
        raise ValueError('discriminator model not supported.')


class Classifier:
    """Classifier backbone
    """
    def tune_params(self, x_train, y_train, x_valid, y_valid):
        raise NotImplementedError

    def fit(self, xs, ys):
        raise NotImplementedError

    def predict(self, xs):
        raise NotImplementedError


class LogReg(Classifier):
    def __init__(self, prob_labels, params=None, seed=None):
        self.prob_labels = prob_labels
        self.model = None
        self.best_params = None
        if params is None:
            params = {
                'solver': ['liblinear'],
                'max_iter': [1000],
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
            }
        self.params = params
        self.n_trials = 10
        if seed is None:
            self.seed = np.random.randint(1e6)
        else:
            self.seed = seed


    def tune_params(self, x_train, y_train, x_valid, y_valid, sample_weights=None, scoring='acc'):
        search_space = self.params

        if self.prob_labels:
            x_train = np.vstack((x_train, x_train))
            weights = np.hstack((1.-y_train, y_train))
            y_train = np.hstack([-np.ones(len(y_train)), np.ones(len(y_train))])
            if sample_weights is not None:
                sample_weights = np.hstack((sample_weights, sample_weights)) * weights
            else:
                sample_weights = weights

        def objective(trial):
            suggestions = {key: trial.suggest_categorical(key, search_space[key]) for key in search_space}

            model = LogisticRegression(**suggestions, random_state=self.seed)
            model.fit(x_train, y_train, sample_weights)            

            ys_pred = model.predict(x_valid)
            
            if scoring == 'acc':
                val_score = accuracy_score(y_valid, ys_pred)
            elif scoring == 'f1':
                val_score = f1_score(y_valid, ys_pred)

            return val_score
        
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)

        self.best_params = study.best_params


    def fit(self, xs, ys, sample_weights=None):
        if self.prob_labels:
            xs = np.vstack((xs, xs))
            weights = np.hstack((1.-ys, ys))
            ys = np.hstack([-np.ones(len(ys)), np.ones(len(ys))])
            if sample_weights is not None:
                sample_weights = np.hstack((sample_weights, sample_weights)) * weights
            else:
                sample_weights = weights

        if self.best_params is not None:
            model = LogisticRegression(**self.best_params, random_state=self.seed)
            model.fit(xs, ys, sample_weight=sample_weights)
            self.model = model
        else:
            raise ValueError('Should perform hyperparameter tuning before fitting')


    def predict(self, xs):
        return self.model.predict(xs)


    def predict_proba(self, xs):
        return self.model.predict_proba(xs)
        

#### unused modules below ####

class LogRegTorch(Classifier):
    def __init__(self, prob_labels, params=None, seed=None):
        self.prob_labels = prob_labels
        self.model = None
        self.best_params = None
        if params is None:
            params = {
                'lr': [1e-4, 1e-3, 1e-2],
                'l2': [1e-4, 1e-3, 1e-2],
                'n_epochs': [50],
                'patience': [3],
                'batch_size': [256]
            }
        self.params = params
        self.n_trials = 20

        if seed is None:
            self.seed = np.random.randint(1e6)
        else:
            self.seed = seed


    def _to_torch_labels(self, ys):
        ys = np.copy(ys)
        ys[ys==-1] = 0
        return ys


    def tune_params(self, x_train, y_train, x_valid, y_valid, sample_weights=None):
        seed = self.seed
        search_space = self.params

        if not self.prob_labels:
            y_train = self._to_torch_labels(y_train)
        y_valid = self._to_torch_labels(y_valid)

        train_dataset = LabeledDataset(x_train, y_train, sample_weights)
        valid_dataset = LabeledDataset(x_valid, y_valid)

        def objective(trial):
            suggestions = {key: trial.suggest_categorical(key, search_space[key]) for key in search_space}
            batch_size = suggestions.pop('batch_size')

            torch.manual_seed(seed)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

            model = LogRegTorchBase(input_dim=500, **suggestions)
            model.fit(train_loader, seed)            

            val_loss = 0.
            for xs_val, ys_val, _ in valid_loader:
                val_loss += F.binary_cross_entropy_with_logits(model(xs_val), ys_val).item()
            val_loss /= len(valid_loader)

            return val_loss
        
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)

        self.best_params = study.best_params
    

    def fit(self, xs, ys, sample_weights=None):
        seed = self.seed
        if not self.prob_labels:
            ys = self._to_torch_labels(ys)
        dataset = LabeledDataset(xs, ys, sample_weights)
        batch_size = self.best_params.pop('batch_size')
        torch.manual_seed(seed)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = LogRegTorchBase(input_dim=500, **self.best_params)
        model.fit(data_loader, seed=seed)
        self.model = model


    def predict(self, xs):
        ys_pred = self.predict_proba(xs)
        ys_pred = np.array([np.random.choice(np.where(y==np.max(y))[0]) for y in ys_pred])
        ys_pred[ys_pred==0] = -1
        return ys_pred


    def predict_proba(self, xs):
        xs = torch.tensor(xs).float()
        ys_pred = self.model.predict_proba(xs)
        ys_pred = np.hstack((1.-ys_pred, ys_pred))
        return ys_pred


class LogRegTorchBase(nn.Module):
    def __init__(self, input_dim, lr, l2, n_epochs, patience):
        super().__init__()
        self.linear_0 = nn.Linear(input_dim, 1)
        self.lr = lr
        self.l2 = l2
        self.n_epochs= n_epochs
        self.patience = patience

    def forward(self, x):
        logit = self.linear_0(x)
        return logit

    def fit(self, train_loader, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)
        
        pre_loss = np.inf
        patience = self.patience
        for epoch in range(self.n_epochs):
            running_loss = 0.
            for xs_tr, ys_tr, weights in train_loader:
                optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(self(xs_tr), ys_tr, reduction='none')
                loss = (loss * weights).mean()
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
            running_loss /= len(train_loader)

            if pre_loss - running_loss < 1e-4:
                patience -=1
            else:
                patience = self.patience
            
            if patience < 0:
                break

    def predict_proba(self, xs):
        logits = self(xs)
        ys_pred = torch.sigmoid(logits).detach().numpy()
        return ys_pred


class LogRegEM(Classifier):
    def __init__(self, params=None, seed=None):
        self.model = None
        self.best_params = None
        if params is None:
            params = {
                'lr': [1e-4, 1e-3, 1e-2],
                'l2': [0.],
                'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
                'n_epochs': [200],
                'patience': [3],
                'batch_size_l': [5],
                'batch_size_u': [256]
            }
        self.params = params
        self.n_trials = 100

        if seed is None:
            self.seed = np.random.randint(1e6)
        else:
            self.seed = seed


    def _to_torch_labels(self, ys):
        ys = np.copy(ys)
        ys[ys==-1] = 0
        return ys


    def tune_params(self, xs_l, ys_l, xs_u, xs_valid, ys_valid, sample_weights=None):
        seed = self.seed
        search_space = self.params

        ys_l = self._to_torch_labels(ys_l)
        ys_valid = self._to_torch_labels(ys_valid)

        labeled_dataset = LabeledDataset(xs_l, ys_l)
        unlabeled_dataset = UnlabeledDataset(xs_u)
        valid_dataset = LabeledDataset(xs_valid, ys_valid)

        def objective(trial):
            suggestions = {key: trial.suggest_categorical(key, search_space[key]) for key in search_space}
            batch_size_l = suggestions.pop('batch_size_l')
            batch_size_u = suggestions.pop('batch_size_u')
            labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size_l, shuffle=True)
            unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size_u, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size_u, shuffle=False)

            model = LogRegEMBase(input_dim=1000, **suggestions)

            val_loss = model.fit(labeled_loader, unlabeled_loader, valid_loader, seed)            

            return val_loss
        
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)

        self.best_params = study.best_params
    

    def fit(self, xs_l, ys_l, xs_u, xs_valid, ys_valid, sample_weights=None):
        seed = self.seed
        ys_l = self._to_torch_labels(ys_l)
        labeled_dataset = LabeledDataset(xs_l, ys_l)
        unlabeled_dataset = UnlabeledDataset(xs_u)
        valid_dataset = LabeledDataset(xs_valid, ys_valid)
        batch_size_l = self.best_params.pop('batch_size_l')
        batch_size_u = self.best_params.pop('batch_size_u')
        labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size_l, shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size_u, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size_u, shuffle=False)
        model = LogRegEMBase(input_dim=1000, **self.best_params)
        model.fit(labeled_loader, unlabeled_loader, valid_loader, seed=seed)
        self.model = model


    def predict(self, xs):
        raise NotImplementedError


    def predict_proba(self, xs):
        xs = torch.tensor(xs).float()
        ys_pred = self.model.predict_proba(xs)
        ys_pred = np.hstack((1.-ys_pred, ys_pred))
        return ys_pred


class LogRegEMBase(nn.Module):
    def __init__(self, input_dim, lr, l2, n_epochs, patience, alpha):
        super().__init__()
        self.linear_0 = nn.Linear(input_dim, 1)
        self.lr = lr
        self.l2 = l2
        self.n_epochs= n_epochs
        self.patience = patience
        self.alpha = alpha

    def forward(self, x):
        logit = self.linear_0(x)
        return logit

    def fit(self, labeled_loader, unlabeled_loader, valid_loader=None, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        alpha = self.alpha
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)
        
        pre_loss = np.inf
        best_val_loss = np.inf
        patience = self.patience
        best_state = None
        for epoch in range(self.n_epochs):
            running_loss = 0.
            n_batch = 0
            for (xs_l, ys_l),  xs_u in zip(labeled_loader, unlabeled_loader):
                optimizer.zero_grad()
                loss_l = F.binary_cross_entropy_with_logits(self(xs_l), ys_l)
                ys_pred_u = torch.sigmoid(self(xs_u))
                loss_u = -(ys_pred_u * torch.log(ys_pred_u+1e-6) + (1-ys_pred_u) * torch.log(1-ys_pred_u+1e-6)).mean()
                loss = loss_l + alpha * loss_u
                running_loss += loss.item()
                n_batch += 1
                loss.backward()
                optimizer.step()

            running_loss /= n_batch

            if pre_loss - running_loss < 1e-4:
                patience -=1
            else:
                pre_loss = running_loss
                best_state = self.state_dict()
                patience = self.patience

            if patience < 0:
                self.load_state_dict(best_state)
                break



        # print(epoch)
            # if valid_loader is not None:
            #     val_loss = 0.
            #     for xs_val, ys_val in valid_loader:
            #         val_loss += F.binary_cross_entropy_with_logits(self(xs_val), ys_val).item()
            #     val_loss /= len(valid_loader)

            #     if val_loss < best_val_loss:
            #         best_val_loss = val_loss
            #         best_state = self.state_dict()
            #         patience = self.patience
            #     else:
            #         patience -= 1
                
            #     if patience < 0:
            #         self.load_state_dict(best_state)
            #         break

        # return best_val_loss
            
            

        if valid_loader is not None:
            val_loss = 0.
            for xs_val, ys_val in valid_loader:
                val_loss += F.binary_cross_entropy_with_logits(self(xs_val), ys_val).item()
            val_loss /= len(valid_loader)

            return val_loss


    def predict_proba(self, xs):
        logits = self(xs)
        ys_pred = torch.sigmoid(logits).detach().numpy()
        return ys_pred


class LabeledDataset(Dataset):
    def __init__(self, xs, ys, weights=None):
        assert len(xs) == len(ys)
        self.xs = torch.tensor(xs).float()
        self.ys = torch.tensor(ys).float().view(-1, 1)
        if weights is not None:
            self.weights = torch.tensor(weights).float().view(-1, 1)
        else:
            self.weights = torch.ones_like(self.ys)
    
    def __getitem__(self, index):
        return (self.xs[index], self.ys[index], self.weights[index])

    def __len__(self):
        return len(self.xs)


class UnlabeledDataset(Dataset):
    def __init__(self, xs):
        self.xs = torch.tensor(xs).float()
    
    def __getitem__(self, index):
        return self.xs[index]

    def __len__(self):
        return len(self.xs)


class SVM(Classifier):
    def __init__(self, params=None):
        self.model = None
        self.best_params = None
        if params is None:
            params = {
                'max_iter': [1000],
                'C':[0.001, 0.01, 0.1, 1, 10, 100],
            }
        self.params = params


    def tune_params(self, x_train, y_train, x_valid, y_valid):
        params = self.params

        xs_train_valid = np.vstack((x_train, x_valid))
        ys_train_valid = np.hstack((y_train, y_valid))
        split_index = [-1 if i < len(x_train) else 0 for i in range(len(xs_train_valid))]
        ps = PredefinedSplit(test_fold=split_index)

        estimator = LinearSVC()
        model = GridSearchCV(estimator, cv=ps, param_grid=params, refit=False)
        model.fit(xs_train_valid, ys_train_valid)
        best_params = model.best_params_
        self.best_params = best_params
    

    def fit(self, xs, ys):
        if self.best_params is not None:
            model = LinearSVC(**self.best_params)
            model.fit(xs, ys)
            self.model = model
        else:
            raise ValueError('Should perform hyperparameter tuning before fitting')


    def predict(self, xs):
        return self.model.predict(xs)