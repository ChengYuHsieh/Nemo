import numpy as np


#### lf utils ####
def compute_coverage(labels):
    coverage  = np.mean(labels != 0)
    return coverage


def compute_gt_precision(labels, ys):
    non_abstain_mask = labels != 0
    precision = np.mean(labels[non_abstain_mask] == ys[non_abstain_mask])
    return precision


def filter_abtain(L, ys):
    non_abstain = (L!=0).any(axis=1)
    L = L[non_abstain]
    ys = ys[non_abstain]
    return L, ys, non_abstain


#### label utils ####
def prob_to_label(ys_pred_prob):
    ys_pred = np.array([np.random.choice(np.where(y==np.max(y))[0]) for y in ys_pred_prob])
    ys_pred[ys_pred==0] = -1
    return ys_pred


def to_onehot(ys):
    ys = np.array(ys).astype(int)
    ys[ys==-1] = 0
    ys_onehot = np.zeros((len(ys), 2))
    ys_onehot[range(len(ys_onehot)), ys] = 1
    return ys_onehot