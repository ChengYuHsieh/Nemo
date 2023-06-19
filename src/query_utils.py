import numpy as np
from scipy import sparse
from tqdm import tqdm
from multiprocessing import Pool
from lf_utils import SentimentLF, compute_gt_precision
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_selection import VarianceThreshold
import yake                
# import spacy
import pdb


class Lexicon(object):
    def __init__(self, keyword_dict):
        self.keyword_dict = keyword_dict

    def get_keywords(self, x):
        keywords = list()
        for token in x:
            if token in self.keyword_dict:
                keywords.append(token)
        return keywords


def build_keyword_dict(xs, dict_size):
    print('buliding keyword list...')
    corpus = [' '.join(doc) for doc in xs]
    # vectorizer = CountVectorizer(max_df=0.1, min_df=20/len(corpus), stop_words='english', max_features=dict_size)
    vectorizer = CountVectorizer(min_df=20, stop_words='english')
    X = vectorizer.fit_transform(corpus).toarray()
    keyword_dict = vectorizer.vocabulary_
    print(f'Lexicon Size: {len(keyword_dict)}')

    return keyword_dict


def cross_entropy(ps, qs):
    ce = -(ps * np.log(np.clip(qs, 1e-8, 1-1e-8)) + (1-ps) * np.log(np.clip(1-qs, 1e-8, 1-1e-8)))
    ce = ce.sum()
    return ce


class LFModel:
    def __init__(self, xs, sentiment_lexicon, pn, kw, dict_size):
        self.xs = xs
        self.lexicon = sentiment_lexicon
        self.pn = pn
        self.kw = kw
        self.dict_size = dict_size
        self.X_pos, self.X_neg, self.vectorizer_pos, self.vectorizer_neg = self.build_X(pn, kw, dict_size) # build the example-pos_keyword matrix
    
    
    def update(self, keyword):
        # remove keyword from X_pos or X_neg
        if keyword in self.vectorizer_pos.vocabulary_:
            idx = self.vectorizer_pos.vocabulary_[keyword]
            self.X_pos[:, idx] = 0.

        if keyword in self.vectorizer_neg.vocabulary_:
            idx = self.vectorizer_neg.vocabulary_[keyword]
            self.X_neg[:, idx] = 0.


    def update_none(self, keywords):
        for keyword in keywords:
            self.update(keyword)


    def compute_lf_prob(self, ys_pred=None):
        X_pos = np.copy(self.X_pos) # example_pos_keyword frequency matrix
        X_neg = np.copy(self.X_neg) # example_neg_keyword frequency matrix

        #### uniform
        # norm = X_pos.sum(axis=1)
        # norm[norm!=0] = 1. / norm[norm!=0]
        # X_lambda_pos_prob = (X_pos.T * norm).T

        # norm = X_neg.sum(axis=1)
        # norm[norm!=0] = 1. / norm[norm!=0]
        # X_lambda_neg_prob = (X_neg.T * norm).T


        #### precision weighted
        # t = 0.01
        t = 1.
        p_neg = ys_pred[:, 0].mean(axis=0)
        p_pos = ys_pred[:, 1].mean(axis=0)

        X_pos = (X_pos != 0).astype(float)
        X_neg = (X_neg != 0).astype(float)

        precision_pos = (X_pos.T * ys_pred[:, 1]).T.sum(axis=0)
        norm = X_pos.sum(axis=0)
        precision_pos[norm!=0] /= norm[norm!=0]

        precision_neg = (X_neg.T * ys_pred[:, 0]).T.sum(axis=0)
        norm = X_neg.sum(axis=0)
        precision_neg[norm!=0] /= norm[norm!=0]

        X_lambda_pos_prob = np.exp(precision_pos / t) * X_pos
        norm = X_lambda_pos_prob.sum(axis=1)
        norm[norm!=0] = 1. / norm[norm!=0]
        X_lambda_pos_prob = (X_lambda_pos_prob.T * norm).T

        X_lambda_neg_prob = np.exp(precision_neg / t) * X_neg
        norm = X_lambda_neg_prob.sum(axis=1)
        norm[norm!=0] = 1. / norm[norm!=0]
        X_lambda_neg_prob = (X_lambda_neg_prob.T * norm).T


        return X_lambda_pos_prob , X_lambda_neg_prob


    def compute_lf_score_cluster(self, label_matrix, disc_model, xs_feature):
        X_pos = np.copy(self.X_pos)
        X_neg = np.copy(self.X_neg)
        X_pos = (X_pos!=0).astype(int)
        X_neg = (X_neg!=0).astype(int)

        pos_lf_centroids = xs_feature.T.dot(X_pos)
        num_coverage = X_pos.sum(axis=0)
        non_abstain = num_coverage != 0
        pos_lf_centroids[:, non_abstain] /= num_coverage[non_abstain] 
        pos_lf_centroids = pos_lf_centroids.T
        
        neg_lf_centroids = xs_feature.T.dot(X_neg)
        num_coverage = X_neg.sum(axis=0)
        non_abstain = num_coverage != 0
        neg_lf_centroids[:, non_abstain] /= num_coverage[non_abstain] 
        neg_lf_centroids = neg_lf_centroids.T

        pred_pos = disc_model.predict_proba(pos_lf_centroids)
        pos_lf_scores = -(pred_pos * np.log(np.clip(pred_pos, 1e-8, 1-1e-8))).sum(axis=1)

        pred_neg = disc_model.predict_proba(neg_lf_centroids)
        neg_lf_scores = -(pred_neg * np.log(np.clip(pred_neg, 1e-8, 1-1e-8))).sum(axis=1)


        return pos_lf_scores, neg_lf_scores 




    def compute_lf_score_moc(self, label_matrix, label_model, disc_model):
        X_pos = np.copy(self.X_pos)
        X_neg = np.copy(self.X_neg)

        num_neg = np.sum(label_matrix == -1, axis=1)
        num_pos = np.sum(label_matrix == 1, axis=1)
        
        # current mv prediction based on "positive label"
        num_nonabstain = num_pos + num_neg
        cur_pred = np.full(label_matrix.shape[0], 0.5)
        cur_pred[num_nonabstain!=0] = num_pos[num_nonabstain!=0] / (num_pos + num_neg)[num_nonabstain!=0]

        # prepare possible updated states
        pos = (num_pos + 1) / (num_pos + num_neg + 1)
        neg = (num_pos) / (num_pos + num_neg + 1)
        non_abstain = (num_pos + num_neg) != 0
        abstain = np.zeros_like(pos).astype(int)
        abstain[non_abstain] = (num_pos)[non_abstain] / (num_pos + num_neg)[non_abstain]

        # updated predictions for pos lfs
        choices = np.hstack([abstain.reshape(-1, 1), pos.reshape(-1, 1)])
        X_pos = X_pos.astype(int)
        X_pos[X_pos!=0] = 1
        new_pred_pos_lfs = np.take_along_axis(choices, X_pos, axis=1)

        # updated prediction for neg lfs
        choices = np.hstack([abstain.reshape(-1, 1), neg.reshape(-1, 1)])
        X_neg = X_neg.astype(int)
        X_neg[X_neg!=0] = 1
        new_pred_neg_lfs = np.take_along_axis(choices, X_neg, axis=1)


        # calculate score for pos lfs
        pos_lf_scores = np.array([cross_entropy(cur_pred, new_pred_pos_lfs[:, j]) for j in range(new_pred_pos_lfs.shape[1])])
        # pos_lf_scores = np.abs(new_pred_pos_lfs.T - cur_pred).sum(axis=1)

        # calculate score for neg lfs
        neg_lf_scores = np.array([cross_entropy(cur_pred, new_pred_neg_lfs[:, j]) for j in range(new_pred_neg_lfs.shape[1])])
        # neg_lf_scores = np.abs(new_pred_neg_lfs.T - cur_pred).sum(axis=1)


        return pos_lf_scores, neg_lf_scores

    

    def compute_lf_score(self, method, xs_score, ys_pred=None):
        X_pos = np.copy(self.X_pos)
        X_neg = np.copy(self.X_neg)

        lambda_pos = (X_pos!=0).astype(float) # shape(num_xs, num_pos_keywords), each column corresponds to the coverage of an positive LF
        lambda_neg = (X_neg!=0).astype(float) # shape(num_xs, num_neg_keywords), each column corresponds to the coverage of an negative LF

        if method == 'sum':
            pos_lf_scores = (lambda_pos.T * xs_score).T.sum(axis=0)
            neg_lf_scores = (lambda_neg.T * xs_score).T.sum(axis=0)
        elif method == 'mean':
            pos_lf_scores = (lambda_pos.T * xs_score).T.sum(axis=0)
            neg_lf_scores = (lambda_neg.T * xs_score).T.sum(axis=0)
            pos_lf_scores[pos_lf_scores!=0] /= lambda_pos.sum(axis=0)[pos_lf_scores!=0]
            neg_lf_scores[neg_lf_scores!=0] /= lambda_neg.sum(axis=0)[neg_lf_scores!=0]
        elif method == 'mean-shift':
            xs_score = xs_score - np.mean(xs_score)
            pos_lf_scores = (lambda_pos.T * xs_score).T.sum(axis=0)
            neg_lf_scores = (lambda_neg.T * xs_score).T.sum(axis=0)
        elif method == 'weighted':
            pos_lf_scores = ((lambda_pos.T * xs_score) *  (2 * ys_pred[:, 1] - 1)).T.sum(axis=0)
            neg_lf_scores = ((lambda_neg.T * xs_score) *  (2 * ys_pred[:, 0] - 1)).T.sum(axis=0)
        elif method == 'weighted-mean':
            pos_lf_scores = ((lambda_pos.T * xs_score) *  (2 * ys_pred[:, 1] - 1)).T.sum(axis=0)
            neg_lf_scores = ((lambda_neg.T * xs_score) *  (2 * ys_pred[:, 0] - 1)).T.sum(axis=0)
            pos_lf_scores[pos_lf_scores!=0] /= lambda_pos.sum(axis=0)[pos_lf_scores!=0]
            neg_lf_scores[neg_lf_scores!=0] /= lambda_neg.sum(axis=0)[neg_lf_scores!=0]
        elif method == 'uncertainty':
            raise NotImplementedError
        else:
            raise NotImplementedError
         
        return pos_lf_scores, neg_lf_scores


    def build_X(self, pn=False, kw=False, dict_size=500):
        """ Build the example-keyword matrix and keyword dictionary.
        One for positive keywords, one for negative keywords.
        """
        xs = self.xs

        if not kw:
            # extract generic keywords without external corpus
            keyword_dict = build_keyword_dict(xs, dict_size=dict_size)
            lexicon = Lexicon(keyword_dict)
            self.lexicon = lexicon

        xs_pos_keywords = list()
        xs_neg_keywords = list()

        if not pn:
            for x in tqdm(xs):
                if not kw:
                    keywords = self.lexicon.get_keywords(x)
                else:
                    pos_keywords = self.lexicon.tokens_with_sentiment(x, 1)
                    neg_keywords = self.lexicon.tokens_with_sentiment(x, -1)
                    keywords = list(pos_keywords) + list(neg_keywords)
                xs_pos_keywords.append(keywords) # Note that there might be repetitive tokens
                xs_neg_keywords.append(keywords) # Note that there might be repetitive tokens
        else:
            if not kw:
                raise ValueError('No externel keyword set provided')
            for x in tqdm(xs):
                xs_pos_keywords.append(self.lexicon.tokens_with_sentiment(x, 1)) # Note that there might be repetitive tokens
                xs_neg_keywords.append(self.lexicon.tokens_with_sentiment(x, -1)) # Note that there might be repetitive tokens

        # build dictionary for positive keywords
        vectorizer_pos = CountVectorizer(preprocessor=lambda x:x, tokenizer=lambda x:x)
        X_pos = vectorizer_pos.fit_transform(xs_pos_keywords).toarray().astype(float)
        
        # build dictionary for negative keywords
        vectorizer_neg = CountVectorizer(preprocessor=lambda x:x, tokenizer=lambda x:x)
        X_neg = vectorizer_neg.fit_transform(xs_neg_keywords).toarray().astype(float)

        return X_pos, X_neg, vectorizer_pos, vectorizer_neg



class ScoringFunction:
    def __init__(self, method):
        self.method = method

    def apply(self, xs_feature=None, label_matrix=None, label_model=None, disc_model=None):
        if self.method == 'random':
            scores = np.ones(len(xs_feature))
        elif self.method == 'abstain':
            scores = (label_matrix == 0).sum(axis=1)
        elif self.method == 'disagreement':
            scores = label_matrix.shape[1] - np.abs(label_matrix.sum(axis=1))
        elif self.method == 'uncertainty_lm':
            ys_pred = label_model.predict_proba(label_matrix)
            scores = -(ys_pred * np.log(np.clip(ys_pred, 1e-8, 1-1e-8))).sum(axis=1)
        elif self.method == 'uncertainty_dm':
            ys_pred = disc_model.predict_proba(xs_feature)
            scores = -(ys_pred * np.log(np.clip(ys_pred, 1e-8, 1-1e-8))).sum(axis=1)
        elif self.method == 'uncertainty_mix':
            ys_pred = label_model.predict_proba(label_matrix) # TODO: consider update label model every query
            scores_lm = -(ys_pred * np.log(np.clip(ys_pred, 1e-8, 1-1e-8))).sum(axis=1)
            ys_pred = disc_model.predict_proba(xs_feature)
            scores_dm = -(ys_pred * np.log(np.clip(ys_pred, 1e-8, 1-1e-8))).sum(axis=1)
            scores = scores_lm * scores_dm

        return scores


class QueryAgent:
    def __init__(self, xs_feature, xs_token, query_method, query_size, rand_state, allow_repeat, qei, aggregate):
        self.xs_feature = xs_feature
        self.xs_token = xs_token
        self.query_method = query_method
        self.query_size = query_size
        self.rand_state = rand_state
        self.allow_repeat = allow_repeat
        self.qei = qei
        self.aggregate = aggregate

        self.queried_idxs = list()  # queried idxs by order
        self.candidate_idxs = set(range(len(xs_feature)))  # candidate idxs from which next query would be chosen

        self.scoring_function = ScoringFunction(self.query_method)


    def warm_start(self):
        cur_query_idxs = self.rand_state.choice(sorted(self.candidate_idxs), size=self.query_size, replace=False) 
        self.update_query_model(cur_query_idxs)

        return cur_query_idxs


    def query(self, label_matrix, label_model=None, lf_model=None, ys_pred=None, use_ys_pred=False, disc_model=None):
        candidate_idxs = np.array(sorted(self.candidate_idxs))
        if self.qei:
            assert lf_model is not None
            xs_score = self.scoring_function.apply(self.xs_feature, label_matrix, label_model, disc_model)
            pos_lf_scores, neg_lf_scores = lf_model.compute_lf_score(self.aggregate, xs_score, ys_pred)

            # pos_lf_scores, neg_lf_scores = lf_model.compute_lf_score_moc(label_matrix, label_model, disc_model)

            # pos_lf_scores, neg_lf_scores = lf_model.compute_lf_score_cluster(label_matrix, disc_model, self.xs_feature)
            
            X_lf_pos_prob , X_lf_neg_prob = lf_model.compute_lf_prob(ys_pred)
            
            if not use_ys_pred or ys_pred is None:
                class_p = np.array([0.5, 0.5])

            xs_pos_expected_score = (X_lf_pos_prob * pos_lf_scores).sum(axis=1)
            xs_neg_expected_score = (X_lf_neg_prob * neg_lf_scores).sum(axis=1)

            xs_expected_score = (class_p * np.vstack([xs_neg_expected_score, xs_pos_expected_score]).T).sum(axis=1)

            scores = xs_expected_score[candidate_idxs]
        else:
            # TODO: maybe avoid confusion between local idxs and global idxs    
            xs_feature = self.xs_feature[candidate_idxs] 
            label_matrix = label_matrix[candidate_idxs]
            scores = self.scoring_function.apply(xs_feature, label_matrix, label_model, disc_model)
        cur_query_idxs = list(self.rand_state.choice(np.where(scores == np.max(scores))[0], size=self.query_size, replace=False)) # subset idxs
        cur_query_idxs = candidate_idxs[cur_query_idxs] # global idxs
        self.update_query_model(cur_query_idxs)

        return cur_query_idxs


    def update_query_model(self, cur_query_idxs):
        self.queried_idxs += list(cur_query_idxs)
        if not self.allow_repeat:
            self.candidate_idxs -= set(cur_query_idxs)