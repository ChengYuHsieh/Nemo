import sys
import os
import argparse
import json
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from data_utils import load_data, SentimentLexicon
from lf_utils import LFAgent
from query_utils import QueryAgent, LFModel
from discriminator import get_discriminator
from label_models import *
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pdb
from implyloss import ImplyLoss


def accuracy_score(ys_pred, ys):
    return np.mean(ys_pred == ys)


def f_score(ys_pred, ys):
    return f1_score(ys, ys_pred)


def auc_score(ys_pred_prob, ys):
    return roc_auc_score(ys, ys_pred_prob[:, 1])


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


def prob_to_label(ys_pred_prob):
    ys_pred = np.array([np.random.choice(np.where(y==np.max(y))[0]) for y in ys_pred_prob])
    ys_pred[ys_pred==0] = -1
    return ys_pred


def to_onehot(ys):
    ys = np.array(ys)
    ys[ys==-1] = 0
    ys_onehot = np.zeros((len(ys), 2))
    ys_onehot[range(len(ys_onehot)), ys] = 1

    return ys_onehot


def run_fs(args, train_dataset, valid_dataset, warmup_dataset, test_dataset):
    print('Running fully-supervised baseline...')
    disc_model = get_discriminator(model_type=args.model_type)

    xs_tr = np.vstack([train_dataset.xs_feature, warmup_dataset.xs_feature])
    ys_tr = np.hstack([train_dataset.ys, warmup_dataset.ys])

    disc_model.tune_params(xs_tr, ys_tr, valid_dataset.xs_feature, valid_dataset.ys)
    disc_model.fit(train_dataset.xs_feature, train_dataset.ys)
    ys_pred = disc_model.predict(test_dataset.xs_feature)

    acc_test = accuracy_score(ys_pred, test_dataset.ys)
    auc_test = auc_score(disc_model.predict_proba(test_dataset.xs_feature), test_dataset.ys)
    f1_test = f_score(ys_pred, test_dataset.ys)

    print('Fully-Supervised Acc: {}'.format(acc_test))
    print('Fully-Supervised AUC: {}'.format(auc_test))
    print('Fully-Supervised F1: {}'.format(f1_test))
    sys.exit(1)


def run_vs(args, valid_dataset, test_dataset):
    xs = valid_dataset.xs_feature
    ys = valid_dataset.ys
    if args.model_type == 'torch':
        raise NotImplementedError
    else:
        params = {
            'solver': ['liblinear'],
            'max_iter': [1000],
            'C': [0.001, 0.01, 0.1, 1, 10, 100]
        }
        model = GridSearchCV(LogisticRegression(random_state=rand_state), params, refit=True)
        model.fit(xs, ys)
        ys_pred_prob = model.predict_proba(test_dataset.xs_feature)
        ys_pred = prob_to_label(ys_pred_prob)

    acc_test = accuracy_score(ys_pred, test_dataset.ys)
    auc_test = auc_score(ys_pred_prob, test_dataset.ys)
    f1_test = f_score(ys_pred, test_dataset.ys)
    print('acc_test:', acc_test)
    print('auc_test:', auc_test)
    print('f1_test:', f1_test)


def train_disc_model(args, xs_tr, ys_tr_soft, ys_tr_hard, xs_tr_unlabeled, valid_dataset, warmup_dataset):
    # prepare discriminator
    seed = np.random.randint(1e6)
    disc_model = get_discriminator(model_type=args.model_type, seed=seed)

    if args.soft_training:  # prepare for soft label training
        if xs_tr is None:
            xs_tr = warmup_dataset.xs_feature
            ys_pred_tr = warmup_dataset.ys
            sample_weights = np.ones(len(warmup_dataset))
        else:
            xs_tr = np.vstack((xs_tr, xs_tr))
            ys_pred_tr = np.hstack((-np.ones(len(xs_tr)//2), np.ones(len(xs_tr)//2)))
            sample_weights = np.hstack((ys_tr_soft[:, 0], ys_tr_soft[:, 1]))
            xs_tr = np.vstack((xs_tr, warmup_dataset.xs_feature))
            ys_pred_tr = np.hstack((ys_pred_tr, warmup_dataset.ys))
            sample_weights = np.hstack((sample_weights, np.ones(len(warmup_dataset))))
    else:
        if xs_tr is None:
            xs_tr = warmup_dataset.xs_feature
            ys_pred_tr = warmup_dataset.ys
            sample_weights = None
        else:
            xs_tr = np.vstack((xs_tr, warmup_dataset.xs_feature))
            ys_pred_tr = np.hstack((ys_tr_hard, warmup_dataset.ys))
            sample_weights = None

    if args.model_type == 'ssl':
        disc_model.tune_params(xs_tr, ys_pred_tr, xs_tr_unlabeled,
                                valid_dataset.xs_feature, valid_dataset.ys)
        disc_model.fit(xs_tr, ys_pred_tr, xs_tr_unlabeled, valid_dataset.xs_feature, valid_dataset.ys)
    else:
        disc_model.tune_params(xs_tr, ys_pred_tr, valid_dataset.xs_feature, valid_dataset.ys, sample_weights)
        disc_model.fit(xs_tr, ys_pred_tr, sample_weights)

    return disc_model


def train_label_model(args, train_dataset, valid_dataset, lf_agent, discard=None):
    # prepare training and validation label matrix
    L_tr = lf_agent.L_tr
    L_val = lf_agent.L_val

    # filter out abstained entries
    L_tr_filtered, ys_tr_filtered, filter_mask_tr = filter_abtain(L_tr, train_dataset.ys)
    L_val_filtered, ys_val_filtered, filter_mask_val = filter_abtain(L_val, valid_dataset.ys)

    xs_tr_filtered = train_dataset.xs_feature[filter_mask_tr]
    xs_tr_unlabeled = train_dataset.xs_feature[~filter_mask_tr]

    # get lf labels
    lf_labels = lf_agent.get_lf_labels()
    anchors = lf_agent.get_anchors()

    # create label model here
    class_balance = [0.87, 0.13] if args.dataset == 'sms' else [0.5, 0.5]
    kwargs = {
        'num_lfs': L_tr_filtered.shape[1],
        'lf_labels': lf_labels,
        'discard': discard,
        'anchors': anchors,
        'class_balance': class_balance
    }

    if L_tr.shape[1] < 3:
        label_model = get_label_model('mv', **kwargs)
    else:
        label_model = get_label_model(args.label_model, **kwargs)

    L_tr_filtered, filter_mask_tr = label_model.fit(L_tr_filtered, L_val_filtered, ys_val_filtered,
            xs_tr=train_dataset.xs_feature[filter_mask_tr], xs_val=valid_dataset.xs_feature[filter_mask_val])

    xs_tr_unlabeled = np.vstack((xs_tr_unlabeled, xs_tr_filtered[~filter_mask_tr]))
    xs_tr_filtered = xs_tr_filtered[filter_mask_tr] # after further discarding
    ys_tr_filtered = ys_tr_filtered[filter_mask_tr] # after further discarding

    ys_pred_tr_filtered_soft = label_model.predict_proba(L_tr_filtered)
    ys_pred_tr_filtered_hard = label_model.predict(L_tr_filtered)

    return (label_model, xs_tr_filtered, ys_tr_filtered, ys_pred_tr_filtered_soft,
           ys_pred_tr_filtered_hard, xs_tr_unlabeled)


def train_implyloss(args, train_dataset, valid_dataset, lf_agent, discard=None):
    # prepare training and validation label matrix
    L_tr = lf_agent.L_tr
    L_val = lf_agent.L_val

    # filter out abstained entries
    L_tr_filtered, ys_tr_filtered, filter_mask_tr = filter_abtain(L_tr, train_dataset.ys)

    # get lf labels
    lf_labels = lf_agent.get_lf_labels()
    anchors = lf_agent.get_anchors()
    anchors_idx = lf_agent.get_anchors_idx()

    # create label model here
    class_balance = [0.87, 0.13] if args.dataset == 'sms' else [0.5, 0.5]
    kwargs = {
        'num_lfs': L_tr_filtered.shape[1],
        'lf_labels': lf_labels,
        'discard': discard,
        'anchors': anchors,
        'anchors_idx': anchors_idx,
        'class_balance': class_balance
    }

    # for implyloss we don't have to filter valid data
    implyloss_model = ImplyLoss(**kwargs)
    implyloss_model.fit(L_tr, L_val, valid_dataset.ys,
                    xs_tr=train_dataset.xs_feature, xs_val=valid_dataset.xs_feature)

    return implyloss_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive Data Programming')
    parser.add_argument('--model-type', type=str, default='logistic')
    parser.add_argument('--query-method', type=str, default='random')
    parser.add_argument('--query-size', type=int, default=1)
    parser.add_argument('--num-query', type=int, default=50)
    parser.add_argument('--train-iter', type=int, default=5)
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--warmup-ratio', type=float, default=0)
    parser.add_argument('--lf-acc', type=float, default=0.5)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--lf-method', type=str, default='sentiment')
    parser.add_argument('--label-model', type=str, default='mv')
    parser.add_argument('--run-fs', action='store_true')
    parser.add_argument('--run-vs', action='store_true')
    parser.add_argument('--runs', type=int, nargs='+', default=range(1))
    parser.add_argument('--dataset', type=str, default='AmazonReview')
    parser.add_argument('--feature', type=str, default='tfidf')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--soft-training', action='store_true')
    parser.add_argument('--qei', action='store_true')
    parser.add_argument('--nopn', action='store_true')
    parser.add_argument('--nokw', action='store_true')
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--noadj', action='store_true')
    parser.add_argument('--use-ys-pred', action='store_true')
    parser.add_argument('--aggregate', type=str, default=None)
    parser.add_argument('--discard', type=str, default=None)
    parser.add_argument('--lexicon', type=int, default=None)
    parser.add_argument('--lf-splits', type=int, default=1)
    parser.add_argument('--multi-lf', action='store_true')

    args = parser.parse_args()

    np.random.seed(args.seed)
    rand_state = np.random.RandomState(args.seed)

    # Load datasets
    train_dataset, valid_dataset, test_dataset, warmup_dataset, vectorizer, scaler = load_data(args.dataset, args.feature, args.test_ratio,
                                                           args.valid_ratio, args.warmup_ratio, rand_state=rand_state)

    print('warm up size: {}'.format(len(warmup_dataset)))

    # scoring = 'f1' if args.dataset == 'sms' else 'logloss'
    class_balance = [0.8, 0.2] if args.dataset == 'sms' else [0.5, 0.5]

    # run fully-supervised baseline
    if args.run_fs:
        run_fs(args, train_dataset, valid_dataset, warmup_dataset, test_dataset)

    # run validation-supervised baseline
    if args.run_vs:
        run_vs(args, valid_dataset, test_dataset)

    # Create sentiment lexicon
    sentiment_lexicon = SentimentLexicon()

    for run in args.runs:
        # Model LF distribution per example
        if args.qei:
            lf_model = LFModel(train_dataset.xs_token, sentiment_lexicon, no_pn=args.nopn, no_kw=args.nokw, dict_size=args.lexicon)
        else:
            lf_model = None

        # set random seed
        np.random.seed(run)
        rand_state = np.random.RandomState(run)

        # Model LF distribution per example
        if args.qei:
            lf_model = LFModel(train_dataset.xs_token, sentiment_lexicon, no_pn=args.nopn, no_kw=args.nokw, dict_size=args.lexicon,
                               ys=train_dataset.ys)
        else:
            lf_model = None

        # Start query interation
        query_agent = QueryAgent(train_dataset.xs_feature, train_dataset.xs_token,
                                 args.query_method, args.query_size, rand_state, False, args.qei, args.noadj, args.aggregate)

        gt_simulation = True if args.dataset in ['sms', 'youtube'] else False
        lf_agent = LFAgent(train_dataset, valid_dataset, sentiment_lexicon, method=args.lf_method, rand_state=rand_state, multi_lf=args.multi_lf, gt_simulation=gt_simulation,
                           vectorizer=vectorizer, scaler=scaler, lf_acc=args.lf_acc)

        label_model = None
        disc_model = None

        history = defaultdict(list)

        for t in range(args.num_query + 1):
            if t % args.train_iter == 0:
                if t == 0 and len(warmup_dataset) == 0:
                    pass
                else:
                    # if label_model is None:
                    #     xs_tr = None
                    #     xs_tr_unlabeled = train_dataset.xs_feature
                    #     ys_pred_tr_soft = None
                    #     ys_pred_tr_hard = None
                    # else:
                    #     pass

                    disc_model = train_implyloss(args, train_dataset, valid_dataset, lf_agent, discard=None)

                    ys_pred_prob = disc_model.predict_proba(test_dataset.xs_feature)
                    ys_pred = prob_to_label(ys_pred_prob)
                    acc_test = accuracy_score(ys_pred, test_dataset.ys)
                    auc_test = auc_score(ys_pred_prob, test_dataset.ys)
                    f1_test = f_score(ys_pred, test_dataset.ys)
                    print('Number of queries: {}; Test Acc: {}'.format(t, acc_test))
                    print('Number of queries: {}; Test AUC: {}'.format(t, auc_test))
                    print('Number of queries: {}; Test F1: {}'.format(t, f1_test))
                    history['test_acc'].append(acc_test)
                    history['test_auc'].append(auc_test)
                    history['test_f1'].append(f1_test)

                    if t == args.num_query:
                        break

            # if len(lf_agent.lfs) == 0 or (args.query_method in ['uncertainty_mix', 'uncertainty_dm'] and disc_model is None):
            if (t // args.train_iter) == 0 or (args.query_method in ['uncertainty_mix', 'uncertainty_dm'] and disc_model is None):
                cur_query_idxs = query_agent.warm_start()
            else:
                # for implyloss specifically
                label_model = disc_model

                L_tr = lf_agent.L_tr
                if disc_model is None:
                    ys_pred = None
                else:
                    ys_pred = disc_model.predict_proba(train_dataset.xs_feature)

                cur_query_idxs = query_agent.query(L_tr, label_model, lf_model, train_dataset.ys, ys_pred=ys_pred,
                                    use_ys_pred=args.use_ys_pred, disc_model=disc_model)

            print('Queried Example: {}'.format(train_dataset.xs_text[cur_query_idxs[0]]))

            lf = lf_agent.create_lf(cur_query_idxs)

            if lf is not None:
                if args.lf_method == 'sentiment':
                    if args.multi_lf:
                        for l in lf:
                            print('lf: {} -> {}'.format(l.keyword, l.label))
                    else:
                        print('lf: {} -> {}'.format(lf.keyword, lf.label))

                L_tr, _ = lf_agent.update_label_matrix(lf)

                coverage = compute_coverage(L_tr[:, -1])
                precision = compute_gt_precision(L_tr[:, -1], train_dataset.ys)
                print('LF coverage: {}; LF precision: {}'.format(coverage, precision))

                if t % args.train_iter == (args.train_iter - 1):
                    discard = args.discard
                else:
                    discard = None

                if args.qei:
                    lf_model.update(lf.keyword)
                else:
                    print('No LF returned')
                    if args.qei and args.update:
                        keywords_rm = train_dataset.xs_token[cur_query_idxs[0]]
                        lf_model.update_none(keywords_rm)

        save_path = './random_warmup_{}_val_{}_lf_{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}split_{}_{}_{}_{}/{}'.format(args.warmup_ratio, args.valid_ratio, args.lf_acc,
                                                                                                                  args.dataset, args.model_type, args.lf_method,
                                                                                                                  args.query_method, 'qei' if args.qei else 'noqei',
                                                                                                                  'cond' if args.use_ys_pred else 'nocond',
                                                                                                                  'nopn' if args.nopn else 'pn', 'nokw' if args.nokw else 'kw',
                                                                args.label_model, 'soft' if args.soft_training else 'hard',
                                                                args.lf_splits, args.aggregate,
                                                                args.discard, args.lexicon,
                                                                'update' if args.update else 'None', run)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        with open(save_path, 'w') as f:
            json.dump(history, f)

