from copy import deepcopy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
import itertools

from label_models import LabelModel, to_snorkel_L

ABSTAIN = -1


class RuleNetwork(nn.Module):
    def __init__(self, input_size, n_rules, hidden_size):
        super(RuleNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size + n_rules, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        z = self.model(x)
        score = torch.sigmoid(z)
        return score


class ClassifierNetwork(nn.Module):
    def __init__(self, input_size, n_class, hidden_size):
        super(ClassifierNetwork, self).__init__()
        # self.model = nn.Sequential(
        #     nn.Linear(input_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.8),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.8),
        #     nn.Linear(hidden_size, n_class),
        # )
        self.model = nn.Linear(input_size, n_class)

    def forward(self, x):
        return self.model(x)


class ImplyLossModel(nn.Module):
    def __init__(self, input_size, n_rules, n_class, hidden_size, q):
        super(ImplyLossModel, self).__init__()
        self.backbone_model = ClassifierNetwork(input_size, n_class, hidden_size)
        self.rule_network = RuleNetwork(input_size, n_rules, hidden_size)
        self.n_rules = n_rules
        self.n_class = n_class
        self.rule_embedding = nn.Parameter(torch.eye(n_rules), requires_grad=False)
        self.q = q

    def get_r_score(self, x):
        covered_features = torch.repeat_interleave(x, self.n_rules, dim=0)
        rule_embedding = self.rule_embedding.repeat(x.size(0), 1)
        concat_feature = torch.cat((covered_features, rule_embedding), dim=1)
        r_score = self.rule_network(concat_feature).view(-1, self.n_rules)
        return r_score

    def forward(self, x, L=None):
        proba = torch.softmax(self.backbone_model(x), dim=1)

        # # Eq 6, optional
        # if L is not None:
        #     covered_mask = torch.sum(L != ABSTAIN, dim=1) > 0
        #     if torch.any(covered_mask):
        #         device = x.device
        #         covered_features = x[covered_mask]
        #         covered_weak_labels_list = L[covered_mask]
        #
        #         r_score = self.get_r_score(covered_features)
        #
        #         fire_mask = (covered_weak_labels_list != ABSTAIN) & (r_score > 0.5)
        #         weak_labels_one_hot = torch.eye(self.n_class).to(device)[covered_weak_labels_list]
        #         score = weak_labels_one_hot * r_score.unsqueeze(2) + (1 - weak_labels_one_hot) * (1 - r_score.unsqueeze(2))
        #         score_mean = torch.sum(score * fire_mask.unsqueeze(2), dim=1) / torch.sum(fire_mask, dim=1, keepdim=True)
        #
        #         proba[covered_mask] += torch.nan_to_num(score_mean, nan=0)

        return torch.softmax(proba, dim=1)

    def calculate_labeled_batch_loss(self, x_exemplar, L_exemplar, y_exemplar):

        r_score = self.get_r_score(x_exemplar)
        non_dia_fire_mask = (L_exemplar != ABSTAIN)
        non_dia_fire_mask[torch.eye(self.n_rules).bool()] = False
        equal_mask = L_exemplar == y_exemplar.unsqueeze(1)

        # Eq (2) first term
        r_score_l = r_score.diagonal()
        loss_phi_1 = F.binary_cross_entropy(r_score_l, torch.ones_like(r_score_l), reduction='sum')

        # Eq (2) second term
        mask = non_dia_fire_mask & (~equal_mask)
        if torch.sum(mask):
            r_score_l = r_score.masked_select(mask)
            loss_phi_2 = F.binary_cross_entropy(r_score_l, torch.zeros_like(r_score_l), reduction='sum')
        else:
            loss_phi_2 = 0.0

        # Eq (2) third term
        mask = non_dia_fire_mask & equal_mask
        if torch.sum(mask):
            r_score_l = r_score.masked_select(mask)
            loss_phi_3 = torch.sum(1 - torch.pow(r_score_l, self.q)) / self.q
        else:
            loss_phi_3 = 0.0

        # Eq (1)
        predict_l = self.backbone_model(x_exemplar)
        loss_theta = F.cross_entropy(predict_l, y_exemplar, reduction='sum')

        loss = loss_theta + loss_phi_1 + loss_phi_2 + loss_phi_3

        return loss

    def calculate_unlabeled_batch_loss(self, x, L):
        batch_size = L.size(0)

        # Eq (4)
        r_score = self.get_r_score(x)

        proba = torch.softmax(self.backbone_model(x), dim=1)
        proba_expand = proba[torch.arange(batch_size).unsqueeze(1), L]

        mask = L != ABSTAIN
        score = 1 - r_score.masked_select(mask) * (1 - proba_expand.masked_select(mask))
        loss = F.binary_cross_entropy(score, torch.ones_like(score))

        return loss


class ImplyLossDataset(Dataset):
    def __init__(self, X, L, ys=None):
        super().__init__()
        self.X = X
        self.L = L
        self.ys = ys

    def __getitem__(self, index):
        if self.ys is None:
            return self.X[index], self.L[index]
        else:
            return self.X[index], self.L[index], self.ys[index]

    def __len__(self):
        return self.L.shape[0]


def sample_batch(loader):
    while True:
        for batch in loader:
            yield batch


class ImplyLoss(LabelModel):
    def __init__(self, **kwargs):
        # self.search_space = kwargs['search_space']
        self.search_space = {
            # 'lr': np.logspace(-4, -1, num=4, base=10),
            'lr': [1e-4, 5e-3, 1e-3],
            # 'weight_decay': np.logspace(-4, -1, num=4, base=10),
            'weight_decay': [0.0],
            'q': [0.2, 0.4, 0.6, 0.8],
            'gamma': [0.2, 0.4, 0.6, 0.8],
        }
        # self.q = 0.2
        # self.gamma = 0.1
        # self.lr = 5e-4
        # self.weight_decay = 0.0

        self.hidden_size = 100
        self.n_steps = 10000
        self.device = torch.device('cuda')
        self.batch_size = 64
        self.evaluation_step = 10
        self.tolerance = 100

        self.n_trials = kwargs.get('n_trials', 100)
        self.best_params = None
        self.model = None
        self.num_lfs = kwargs['num_lfs']
        self.anchors = kwargs['anchors']
        self.anchors_idx = kwargs['anchors_idx']
        self.lf_labels = kwargs['lf_labels']
        self.kwargs = kwargs

    def fit_(self, L_tr, L_val, ys_val, xs_tr, xs_val, suggesions: dict):
        gamma = suggesions['gamma']
        q = suggesions['q']
        weight_decay = suggesions['weight_decay']
        lr = suggesions['lr']

        hidden_size = self.hidden_size
        batch_size = self.batch_size
        n_rules = self.num_lfs
        n_steps = self.n_steps
        device = self.device
        evaluation_step = self.evaluation_step

        n_class = len(np.unique(ys_val))
        input_size = xs_tr.shape[1]

        L_tr = to_snorkel_L(L_tr)
        L_val = to_snorkel_L(L_val)
        ys_val = to_snorkel_L(ys_val)

        x_exemplar = torch.Tensor(self.anchors).double().to(device)
        L_exemplar = torch.Tensor(L_tr[self.anchors_idx]).long().to(device)
        y_exemplar = torch.Tensor(to_snorkel_L(np.array(self.lf_labels))).long().to(device)

        mask = (L_tr != ABSTAIN).any(axis=1)
        mask[self.anchors_idx] = 0

        L_tr = L_tr[mask]
        xs_tr = xs_tr[mask]

        unlabeled_dataset = ImplyLossDataset(xs_tr, L_tr)
        dataset_val = ImplyLossDataset(xs_val, L_val, ys_val)

        model = ImplyLossModel(
            input_size=input_size,
            n_rules=n_rules,
            n_class=n_class,
            hidden_size=hidden_size,
            q=q
        )
        self.model = model.double().to(device)

        data_loader_tr = sample_batch(DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True))
        data_loader_val = DataLoader(dataset_val, batch_size=L_val.shape[0], shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        step = 0
        pre_val_loss = np.inf
        best_params = None
        last_step_log = {}
        best_step = -1
        best_acc = -1
        with trange(n_steps, desc="[TRAIN] ImplyLoss", unit="steps", ncols=200, position=0, leave=True, disable=True) as pbar:
            while step < n_steps:
                step += 1

                model.train()
                optimizer.zero_grad()

                unlabeled_batch = next(data_loader_tr)

                batch_x, batch_L = unlabeled_batch
                loss_u = model.calculate_unlabeled_batch_loss(batch_x.to(device), batch_L.to(device))
                loss_l = model.calculate_labeled_batch_loss(x_exemplar, L_exemplar, y_exemplar)
                loss = loss_l + gamma * loss_u

                loss.backward()
                optimizer.step()

                if step % evaluation_step == 0:
                    model.eval()
                    val_loss = 0.
                    correct = 0
                    for batch_x_val, batch_L_val, batch_ys_val in data_loader_val:
                        batch_ys_val = batch_ys_val.to(device)
                        ys_pred = model(batch_x_val.to(device), batch_L_val.to(device))
                        correct += torch.sum(ys_pred.argmax(dim=1) == batch_ys_val)
                        val_loss += F.cross_entropy(ys_pred, batch_ys_val).item()
                    val_loss /= len(data_loader_val)
                    acc = correct.item() / len(ys_val)

                    if val_loss < pre_val_loss:
                    # if acc > best_acc:
                        pre_val_loss = val_loss
                        best_params = deepcopy(self.model.state_dict())
                        tolerance = self.tolerance
                        best_step = step
                        best_acc = acc
                    else:
                        tolerance -= 1

                    if tolerance <= 0:
                        self.model.load_state_dict(best_params)
                        break

                    last_step_log.update({
                        'val_loss'     : val_loss,
                        'val_acc'     : acc,
                        'best_val_loss': pre_val_loss,
                        'best_val_acc': best_acc,
                        'best_step'     : best_step,
                    })

                last_step_log['loss'] = loss.item()
                pbar.update()
                pbar.set_postfix(ordered_dict=last_step_log)

        return pre_val_loss

    def search(self, L_tr, L_val, ys_val, xs_tr, xs_val):
        search_space = self.search_space


        all_grids = list(itertools.product(*search_space.values()))
        param_names = search_space.keys()

        best_metric = 1e5
        best_trial = -1
        best_model = None
        best_paras = None
        print(f'# of grids: {len(all_grids)}')
        for i, grid in enumerate(all_grids):
            suggestions = dict(zip(param_names, grid))
            try:
                valid_metric = self.fit_(L_tr, L_val, ys_val, xs_tr, xs_val, suggestions)
                if valid_metric < best_metric:
                    best_trial = i
                    best_metric = valid_metric
                    best_model = {k: v.cpu() for k, v in self.model.state_dict().items()}
                    best_paras = suggestions
                print(f'Trial-{i}: {suggestions}\tvalid metric={valid_metric} | best trial-{best_trial}\tvalid metric={best_metric}')
            except Exception as e:
                print(e)
                print(f'Trial-{i}: failed.')
            # if i>0:
            #     break

        n_class = len(np.unique(ys_val))
        input_size = xs_tr.shape[1]
        self.model = ImplyLossModel(
            input_size=input_size,
            n_rules=self.num_lfs,
            n_class=n_class,
            hidden_size=self.hidden_size,
            q=best_paras['q'],
        ).double()
        self.model.load_state_dict(best_model)
        self.model.to(self.device)

    def fit(self, L_tr, L_val, ys_val, xs_tr, xs_val):
        self.search(L_tr, L_val, ys_val, xs_tr, xs_val)

    def predict_proba(self, x, L=None, batch_size=-1):
        if L is None:
            x = torch.Tensor(x).to(self.device).double()
            proba = self.model(x)
            return proba.cpu().detach().numpy()

        L = to_snorkel_L(L)
        dataset = ImplyLossDataset(x, L)
        if batch_size == -1:
            batch_size = L.shape[0]
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        with torch.no_grad():
            device = self.device
            probas = []

            for batch_x_val, batch_L_val in data_loader:
                proba = self.model(batch_x_val.to(device), batch_L_val.to(device))
                probas.append(proba.cpu().detach().numpy())

        return np.vstack(probas)

    def predict(self, x, L=None):
        ys_pred = self.predict_proba(x, L)
        # breaking ties by random selection
        ys_pred = np.array([np.random.choice(np.where(y == np.max(y))[0]) for y in ys_pred])
        ys_pred[ys_pred == 0] = -1
        return ys_pred
