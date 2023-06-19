import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
from lime.lime_text import LimeTextExplainer
from utils import *
import pdb


class SentimentLF:
    def __init__(self, keyword, label, anchor=None, anchor_id=None):
        self.keyword = keyword
        self.label = label
        self.anchor = anchor
        self.anchor_id = anchor_id

    def apply(self, sentence):
        if self.keyword in sentence:
            return self.label
        else:
            return 0
    

class LabelLF:
    def __init__(self, sentence, label, anchor=None, anchor_id=None):
        self.sentence = sentence
        self.label = label
        self.anchor = anchor
        self.anchor_id = anchor_id

    def apply(self, sentence):
        if sentence == self.sentence:
            return self.label
        else:
            return 0



class LFAgent:
    def __init__(self, train_dataset, valid_dataset, sentiment_lexicon, method='sentiment', rand_state=None, lf_acc=None, lf_simulate=None):
        self.xs_rawtext_tr = train_dataset.xs_text
        self.xs_feature_tr = train_dataset.xs_feature
        self.xs_text_tr = train_dataset.xs_token
        self.ys_tr = train_dataset.ys
        self.L_tr = None

        self.xs_rawtext_val = valid_dataset.xs_text
        self.xs_feature_val = valid_dataset.xs_feature
        self.xs_text_val = valid_dataset.xs_token
        self.ys_val = valid_dataset.ys
        self.L_val = None

        self.sentiment_lexicon = sentiment_lexicon
        self.lfs = list()
        self.keywords = set()
        
        self.method = method
        self.rand_state = rand_state

        self.lf_acc = lf_acc
        self.lf_simulate = lf_simulate

        if lf_simulate == 'expl':
            print('Preparing user model...')

            vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(strip_accents='ascii', stop_words='english')
            train_vectors = vectorizer.fit_transform(train_dataset.xs_text)
            tokenizer = vectorizer.build_tokenizer()

            # svd = TruncatedSVD(n_components=500, n_iter=20, random_state=42)
            # train_vectors = svd.fit_transform(train_vectors)

            # params = {
            #             'solver': ['liblinear'],
            #             'max_iter': [1000],
            #             'C': [0.001, 0.01, 0.1, 1, 10, 100]
            #         }

            # params = {
                        # 'n_estimators': [100, 250, 500],
                    # }
            # model = GridSearchCV(LogisticRegression(random_state=0), params, refit=True)
            # model = GridSearchCV(RandomForestClassifier(random_state=0), params, refit=True)
            model = RandomForestClassifier(random_state=0)
            model.fit(train_vectors, train_dataset.ys)
            # model = model.best_estimator_
            # pipeline = make_pipeline(vectorizer, svd, model)
            pipeline = make_pipeline(vectorizer, model)
            print(f'user model valid acc: {pipeline.score(self.xs_rawtext_val, self.ys_val)}')

            self.user_model = pipeline
            self.tokenizer = tokenizer


    def create_lf(self, cur_query_idxs):
        # select one queried point to create LF
        idx = self.rand_state.choice(cur_query_idxs, size=1)[0]
        x = self.xs_text_tr[idx]
        y = self.ys_tr[idx]

        if self.method == 'sentiment':
            if self.lf_simulate == 'expl':
                explainer = LimeTextExplainer(split_expression=self.tokenizer, random_state=0)
                try:
                    exp = explainer.explain_instance(self.xs_rawtext_tr[idx], self.user_model.predict_proba, labels=((y+1)//2,),
                                                     num_features=30, num_samples=20000)
                except:
                    return None
                tokens = list()
                p = list()
                for w, score in exp.as_list(label=(y+1)//2):
                    if score > 0 and w.lower() not in self.keywords:
                        lf = SentimentLF(keyword=w.lower(), label=y, anchor=self.xs_feature_tr[idx], anchor_id=idx)
                        labels_tr = np.array(list(map(lf.apply, self.xs_text_tr)))
                        precision = compute_gt_precision(labels_tr, self.ys_tr)
                        if precision > self.lf_acc:
                            tokens.append(w.lower())
                            p.append(score)
                        else:
                            pass

                if len(tokens) == 0:
                    lf = None
                else:
                    p = np.array(p)
                    p = p / p.sum()
                    token = tokens[self.rand_state.choice(range(len(tokens)), p=p)]
                    lf = SentimentLF(keyword=token, label=y, anchor=self.xs_feature_tr[idx], anchor_id=idx)
            elif self.lf_simulate == 'lexicon':
                tokens = self.sentiment_lexicon.tokens_with_sentiment(x, y)
                tokens = [token for token in tokens if token not in self.keywords] # avoid creating duplicate LFs, a natural assumption simulating real users
                if len(tokens) == 0:
                    lf = None
                else:
                    while len(tokens) > 0:
                        token = self.rand_state.choice(tokens, size=1)[0]
                        lf = SentimentLF(keyword=token, label=y, anchor=self.xs_feature_tr[idx], anchor_id=idx)
                        labels_tr = np.array(list(map(lf.apply, self.xs_text_tr)))
                        precision = compute_gt_precision(labels_tr, self.ys_tr)
                        if precision > self.lf_acc:
                            break
                        else:
                            tokens.remove(token)

                    if len(tokens) == 0:
                        lf = None
            elif self.lf_simulate == 'acc':
                tokens = [token for token in x if token not in self.keywords]
                candidate_lfs = list()
                coverages = list()
                for token in tokens:
                    lf = SentimentLF(keyword=token, label=y, anchor=self.xs_feature_tr[idx], anchor_id=idx)
                    labels_tr = np.array(list(map(lf.apply, self.xs_text_tr)))
                    precision = compute_gt_precision(labels_tr, self.ys_tr)
                    coverage = compute_coverage(labels_tr)
                    if precision > self.lf_acc:
                        candidate_lfs.append(lf)
                        coverages.append(coverage)
                
                if len(candidate_lfs) == 0:
                    lf = None
                else:
                    coverages = np.array(coverages)
                    p = coverages / coverages.sum()
                    lf = candidate_lfs[self.rand_state.choice(range(len(candidate_lfs)), p=p)]

            else:
                raise NotImplementedError()

        elif self.method == 'label':
            lf = LabelLF(sentence=x, label=y, anchor=self.xs_feature_tr[idx], anchor_id=idx)
        else:
            raise NotImplementedError()

        return lf


    def update(self, lf):
        self.lfs.append(lf)
        if self.method == 'sentiment':
            self.keywords.add(lf.keyword)

        labels_tr = np.array(list(map(lf.apply, self.xs_text_tr))).reshape(-1, 1)
        labels_val = np.array(list(map(lf.apply, self.xs_text_val))).reshape(-1, 1)

        if self.L_tr is None:
            assert self.L_val is None
            self.L_tr = labels_tr
            self.L_val = labels_val
        else:
            assert self.L_tr.shape[1] == self.L_val.shape[1]
            self.L_tr = np.hstack((self.L_tr, labels_tr))
            self.L_val = np.hstack((self.L_val, labels_val))

        return self.L_tr, self.L_val


    def get_anchors(self):
        anchors = [lf.anchor for lf in self.lfs]
        return anchors


    def get_anchors_idx(self):
        anchor_idx = [lf.anchor_id for lf in self.lfs]
        return anchor_idx


    def get_lf_labels(self):
        lf_labels = [lf.label for lf in self.lfs]
        return lf_labels


#### not used modules below ####

class Encoder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def fit(self, X):
        pass

    def transform(self, X):
        return self.model.encode(X)