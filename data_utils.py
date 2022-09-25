import os
import sys
import json
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import pdb


def create_bert_vector(raw_texts, save_path):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(raw_texts)
    np.save(save_path, embeddings)

    return embeddings


def tr_val_te_split(xs, ys, test_ratio, valid_ratio, rand_state):
    xs = np.array(xs)
    ys = np.array(ys)
    assert len(xs) == len(ys)
    N = len(xs)

    permuted_idxs = rand_state.permutation(N)
    num_test = int(N * test_ratio)
    num_valid = int(N * valid_ratio)

    train_idxs, test_idxs = permuted_idxs[:-num_test], permuted_idxs[-num_test:]
    # num_valid = int(len(train_idxs) * valid_ratio)
    train_idxs, valid_idxs = train_idxs[:-num_valid], train_idxs[-num_valid:]

    return (xs[train_idxs], ys[train_idxs], xs[valid_idxs], ys[valid_idxs],
            xs[test_idxs], ys[test_idxs], train_idxs, valid_idxs, test_idxs)


def load_data(data_root, dataset_name, feature, test_ratio, valid_ratio, warmup_ratio, rand_state):
    # load raw sentences and labels
    if dataset_name == 'AmazonReview':
        dataset = AmazonReviewDataset(data_root=os.path.join(data_root, 'AmazonReview'))
    elif dataset_name == 'IMDB':
        dataset = IMDBDataset(data_root=os.path.join(data_root, 'aclImdb'))
    elif dataset_name == 'SST':
        dataset = SSTDataset(data_root=os.path.join(data_root, 'SST-2'))
    elif dataset_name == 'Yelp':
        dataset = YelpDataset(data_root=os.path.join(data_root, 'yelp_review_polarity_csv'))
    elif dataset_name == 'sms':
        dataset = SMSDataset(data_root=os.path.join(data_root, 'sms'))
    elif dataset_name == 'bios':
        dataset = BiosDataset(data_root=os.path.join(data_root, 'bios'))
    elif dataset_name == 'agnews':
        dataset = AGNewsDataset(data_root=os.path.join(data_root, 'agnews'), rand_state=rand_state)
    elif dataset_name == 'yahoo':
        dataset = YahooDataset(data_root=os.path.join(data_root, 'yahoo'), rand_state=rand_state)
    elif dataset_name == 'youtube':
        dataset = YoutubeDataset(data_root=os.path.join(data_root, 'spam/data'))
    else:
        raise ValueError('Dataset not supported.')

    raw_texts = dataset.raw_texts
    labels = dataset.labels

    (xs_text_tr, ys_tr, xs_text_val, ys_val, 
    xs_text_te, ys_te, train_idxs, valid_idxs, test_idxs) = tr_val_te_split(raw_texts, labels, test_ratio, valid_ratio, rand_state)

    # create tokenized texts for LF labeling use
    count_vectorizer = CountVectorizer(strip_accents='ascii')
    count_vectorizer.fit(xs_text_tr)
    analyzer = count_vectorizer.build_analyzer()
    xs_token_tr = np.array([analyzer(text) for text in xs_text_tr], dtype='object')
    xs_token_val = np.array([analyzer(text) for text in xs_text_val], dtype='object')
    xs_token_te = np.array([analyzer(text) for text in xs_text_te], dtype='object')

    # create features (independent of the above tokenization process)
    if feature == 'tfidf':
        tfidf_vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words='english', max_df=0.9, max_features=1000)
        xs_feature_tr = tfidf_vectorizer.fit_transform(xs_text_tr).toarray()
        xs_feature_val = tfidf_vectorizer.transform(xs_text_val).toarray()
        xs_feature_te = tfidf_vectorizer.transform(xs_text_te).toarray()

        scaler = StandardScaler()
        xs_feature_tr = scaler.fit_transform(xs_feature_tr)
        xs_feature_val = scaler.transform(xs_feature_val)
        xs_feature_te = scaler.transform(xs_feature_te)

    elif feature == 'embedding':
        vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1), analyzer='word')
        xs_feature_tr = vectorizer.fit_transform(xs_text_tr)
        xs_feature_val = vectorizer.transform(xs_text_val)
        xs_feature_te = vectorizer.transform(xs_text_te)

        n, m = xs_feature_tr.shape

        svd = TruncatedSVD(n_components=500, n_iter=5, random_state=0)
        xs_feature_tr = svd.fit_transform(xs_feature_tr).astype(float)
        xs_feature_val = svd.transform(xs_feature_val).astype(float)
        xs_feature_te = svd.transform(xs_feature_te).astype(float)

        # scaler = StandardScaler()
        # xs_feature_tr = scaler.fit_transform(xs_feature_tr)
        # xs_feature_val = scaler.transform(xs_feature_val)
        # xs_feature_te = scaler.transform(xs_feature_te)

    elif feature == 'bert':
        saved_file = os.path.join(data_root, 'embeddings', dataset_name, 'bert.npy')
        if os.path.exists(saved_file):
            embeddings = np.load(saved_file)
        else:
            save_path = os.path.join(data_root, 'embeddings', dataset_name, 'bert.npy')
            embeddings = create_bert_vector(raw_texts, save_path)

        xs_feature_tr = embeddings[train_idxs]
        xs_feature_val = embeddings[valid_idxs]
        xs_feature_te = embeddings[test_idxs]

    else:
        raise ValueError('Feature representation not supported.')

    num_train = len(ys_tr)
    if warmup_ratio > 1:
        num_warmup = int(warmup_ratio)
    else:
        num_warmup = int(num_train * warmup_ratio)

    permuted_idxs = rand_state.permutation(num_train) 
    warmup_idxs, train_idxs = permuted_idxs[:num_warmup], permuted_idxs[num_warmup:]

    xs_text_wu, xs_token_wu, xs_feature_wu, ys_wu = xs_text_tr[warmup_idxs], xs_token_tr[warmup_idxs], xs_feature_tr[warmup_idxs], ys_tr[warmup_idxs]
    xs_text_tr, xs_token_tr, xs_feature_tr, ys_tr = xs_text_tr[train_idxs], xs_token_tr[train_idxs], xs_feature_tr[train_idxs], ys_tr[train_idxs]

    train_dataset = SentimentDataset(xs_text_tr, xs_token_tr, xs_feature_tr, ys_tr)
    valid_dataset = SentimentDataset(xs_text_val, xs_token_val, xs_feature_val, ys_val)
    test_dataset = SentimentDataset(xs_text_te, xs_token_te, xs_feature_te, ys_te)
    warmup_dataset = SentimentDataset(xs_text_wu, xs_token_wu, xs_feature_wu, ys_wu)

    return train_dataset, valid_dataset, test_dataset, warmup_dataset

    
class SentimentDataset:
    def __init__(self, xs_text, xs_token, xs_feature, ys):
        assert np.all(np.array([len(xs_text), len(xs_token), len(xs_feature)]) == len(ys))
        self.xs_text = xs_text
        self.xs_token = xs_token
        self.xs_feature = xs_feature
        self.ys = ys

    def __len__(self):
        return len(self.ys)


class SSTDataset:
    def __init__(self, data_root, subsample_size=20000):
        self.data_root = data_root
        self.subsample_size = subsample_size
        self.raw_texts, self.labels = self.build_dataset()


    def build_dataset(self):
        data_file = os.path.join(self.data_root, 'train.tsv')
        with open(data_file) as f:
            lines = f.readlines()
        lines = lines[1:] # drop the headers
        lines = [line.rstrip().split('\t') for line in lines]
        lines = np.array(lines, 'object')
        # subsample
        lines = lines[np.random.permutation(len(lines))[:self.subsample_size]]
        raw_texts = lines[:, 0]
        labels = lines[:, 1].astype(int)
        labels = labels * 2 - 1

        return raw_texts, labels


class SMSDataset:
    def __init__(self, data_root):
        self.data_root = data_root
        self.raw_texts, self.labels = self.build_dataset()

    def build_dataset(self):
        data_file = os.path.join(self.data_root, 'spam.csv')
        df = pd.read_csv(data_file, sep=',', header=0, encoding='latin-1').to_numpy()[:, :2]

        raw_texts = df[:, 1]
        labels = df[:, 0]
        labels[labels=='ham'] = -1
        labels[labels=='spam'] = 1
        labels = labels.astype(int)

        return raw_texts, labels


class BiosDataset:
    def __init__(self, data_root):
        self.data_root = data_root
        self.raw_texts, self.labels = self.build_dataset()

    def build_dataset(self):
        data_file = os.path.join(self.data_root, 'professor_teacher.csv')
        # data_file = os.path.join(self.data_root, 'painter_architect.csv')
        # data_file = os.path.join(self.data_root, 'journalist_photographer.csv')
        df = pd.read_csv(data_file, sep=',', header=0).to_numpy()[:, :2]

        raw_texts = df[:, 0]
        labels = df[:, 1]
        labels = labels.astype(int)

        return raw_texts, labels


class YoutubeDataset:
    def __init__(self, data_root):
        self.data_root = data_root
        self.raw_texts, self.labels = self.build_dataset()

    def build_dataset(self):
        files = ['Youtube01-Psy.csv', 'Youtube02-KatyPerry.csv', 'Youtube03-LMFAO.csv',
                 'Youtube04-Eminem.csv', 'Youtube05-Shakira.csv']
        df_all = None
        for f in files:
            data_file = os.path.join(self.data_root, f)
            df = pd.read_csv(data_file, sep=',', header=0)
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all, df])

        df = df_all.to_numpy()
        raw_texts = df[:, 3]
        labels = df[:, 4]
        labels = labels.astype(int)
        labels[labels==0] = -1

        return raw_texts, labels


class AGNewsDataset:
    def __init__(self, data_root, rand_state):
        self.data_root = data_root
        self.rand_state = rand_state
        self.raw_texts, self.labels = self.build_dataset()

    def build_dataset(self):
        data_file = os.path.join(self.data_root, 'train.csv')
        df = pd.read_csv(data_file, sep=',', header=0).to_numpy()

        raw_texts = df[:, 2]
        labels = df[:, 0].astype(int)

        class_0 = 3
        class_1 = 4

        sport_idx = labels == class_0
        sci_idx = labels == class_1

        raw_texts_sport = raw_texts[sport_idx]
        labels_sport = labels[sport_idx]

        raw_texts_sci = raw_texts[sci_idx]
        labels_sci = labels[sci_idx]

        raw_texts_all = np.hstack([raw_texts_sport, raw_texts_sci])
        labels_all = np.hstack([labels_sport, labels_sci])

        idx_permutation = self.rand_state.permutation(len(labels_all))
        raw_texts = raw_texts_all[idx_permutation]
        labels = labels_all[idx_permutation]
        labels[labels==class_0] = -1
        labels[labels==class_1] = 1

        return raw_texts, labels


class YahooDataset:
    def __init__(self, data_root, rand_state):
        self.data_root = data_root
        self.rand_state = rand_state
        self.raw_texts, self.labels = self.build_dataset()

    def build_dataset(self):
        data_file = os.path.join(self.data_root, 'train.csv')
        df = pd.read_csv(data_file, sep=',', header=None)

        df[4] = df[1].astype(str) + ' ' + df[2].astype(str) + ' ' + df[3].astype(str)

        df = df.to_numpy()

        raw_texts = df[:, 4]
        labels = df[:, 0].astype(int)

        class_0 = 2
        class_1 = 6

        class_0_idx = labels == class_0
        class_1_idx = labels == class_1

        raw_texts_0 = raw_texts[class_0_idx]
        labels_0 = labels[class_0_idx]

        raw_texts_1 = raw_texts[class_1_idx]
        labels_1 = labels[class_1_idx]

        raw_texts_all = np.hstack([raw_texts_0, raw_texts_1])
        labels_all = np.hstack([labels_0, labels_1])

        idx_permutation = self.rand_state.permutation(len(labels_all))[:30000]
        raw_texts = raw_texts_all[idx_permutation]
        labels = labels_all[idx_permutation]
        labels[labels==class_0] = -1
        labels[labels==class_1] = 1

        return raw_texts, labels


class YelpDataset:
    def __init__(self, data_root, subsample_size=25000):
        self.data_root = data_root
        self.subsample_size = subsample_size
        self.raw_texts, self.labels = self.build_dataset()

    def build_dataset(self):
        data_file = os.path.join(self.data_root, 'train.csv')
        df = pd.read_csv(data_file, sep=',', header=None).to_numpy()
        # subsample
        df_subsampled = df[np.random.permutation(len(df))[:self.subsample_size]]
        raw_texts = df_subsampled[:, 1]
        labels = df_subsampled[:, 0].astype(int)
        labels = (labels - 1) * 2 - 1

        return raw_texts, labels


class AmazonReviewDataset:
    """ AmazonReview Dataset for Sentiment Analysis (Polarity)
    """
    def __init__(self, data_root, subsample_size=500, pos_rating_threshold=5, neg_rating_threshold=1,
                 saved_path=None, random_state=None):
        self.data_root = data_root
        self.subsample_size = subsample_size
        self.pos_rating_threshold = pos_rating_threshold
        self.neg_rating_threshold = neg_rating_threshold
        self.saved_path = saved_path
        self.random_state = random_state
        
        # create dataset with raw text
        df = None
        if saved_path is None:
            # print('looking for "amazon_review.pkl"...')
            if os.path.exists(f'{data_root}/amazon_review.pkl'):
                # print('loading raw text dataset from {}'.format('"./amazon_review.pkl"'))
                df = pd.read_pickle(f'{data_root}/amazon_review.pkl')
            else:
                print('building raw text dataset...')
                df = self.build_dataset()
                df.to_pickle('./amazon_review.pkl')
        else:
            print('loading raw text dataset...')
            df = pd.read_pickle(saved_path)
        
        # extract raw setences and labels
        self.df = df
        self.raw_texts = df['reviewText'].to_numpy()
        self.labels =  df['sentiment'].to_numpy()
    

    def parse_file(self, path, category):
        print('Processing {}...'.format(category))
        data = list()
        with gzip.open(path) as f:
            for line in f:
                record = json.loads(line)
                # only keep records with rating above/below specified thresholds
                if (record['overall'] >= self.pos_rating_threshold or
                    record['overall'] <= self.neg_rating_threshold):
                    record['category'] = category
                    data.append(record)
        df = pd.DataFrame.from_records(data)

        # keep only relevant features
        df = df[['reviewText', 'category', 'overall']]

        # partition reviews into positive/negative sentiments
        df_pos = df[df['overall'] >= self.pos_rating_threshold]
        df_neg = df[df['overall'] <= self.neg_rating_threshold]

        # subsample data from positive/negative subsets
        df_pos = df_pos.sample(n=self.subsample_size, replace=False, random_state=self.random_state)
        df_pos['sentiment'] = 1
        df_neg = df_neg.sample(n=self.subsample_size, replace=False, random_state=self.random_state)
        df_neg['sentiment'] = -1
        df = pd.concat([df_pos, df_neg], axis=0)

        return df


    def build_dataset(self):
        files = os.listdir(self.data_root)
        paths = [os.path.join(self.data_root, file) for file in files]
        categories = [file.split('.')[0].lower()[8:-2] for file in files]
        print('Categories: {}'.format(categories))

        dfs = list()
        for path, category in tqdm(zip(paths, categories), total=len(paths)):
            df = self.parse_file(path, category)
            if df is not None:
                dfs.append(df)
        df = pd.concat(dfs, axis=0)

        return df


class IMDBDataset:
    def __init__(self, data_root):
        self.data_root = os.path.join(data_root, 'train')
        self.pos_dir = os.path.join(self.data_root, 'pos')
        self.neg_dir = os.path.join(self.data_root, 'neg')

        raw_texts, labels = self.build_dataset()

        self.raw_texts = np.array(raw_texts, dtype='object')
        self.labels = np.array(labels)

    def build_dataset(self):
        pos_files = os.listdir(self.pos_dir)
        pos_paths = [os.path.join(self.pos_dir, file) for file in pos_files]
        pos_texts = list()
        for path in pos_paths:
            with open(path) as f:
                line = f.readline().rstrip()
                pos_texts.append(line)

        neg_files = os.listdir(self.neg_dir)
        neg_paths = [os.path.join(self.neg_dir, file) for file in neg_files]
        neg_texts = list()
        for path in neg_paths:
            with open(path) as f:
                line = f.readline().rstrip()
                neg_texts.append(line)

        raw_texts = pos_texts + neg_texts
        labels = [1] * len(pos_texts) + [-1] * len(neg_texts)

        return raw_texts, labels


class SentimentLexicon:
    def __init__(self, data_root):
        pos_words_file = os.path.join(data_root, 'opinion-lexicon-English/positive-words.txt')
        neg_words_file = os.path.join(data_root, 'opinion-lexicon-English/negative-words.txt')

        pos_tokens = list()
        neg_tokens = list()
        with open(pos_words_file, encoding='ISO-8859-1') as f:
            for i, line in enumerate(f):
                if i >= 30:
                    token = line.rstrip()
                    pos_tokens.append(token)
        with open(neg_words_file, encoding='ISO-8859-1') as f:
            for i, line in enumerate(f):
                if i >= 31:
                    token = line.rstrip()
                    neg_tokens.append(token)

        token_sentiment = {token: 1 for token in pos_tokens}
        token_sentiment.update({token: -1 for token in neg_tokens})

        self.pos_tokens = pos_tokens
        self.neg_tokens = neg_tokens
        self.token_sentiment = token_sentiment


    def tokens_to_sentiments(self, tokens):
        """Return sentiments of tokens in a sentence
        """
        sentiments = np.array([self.token_sentiment.get(token, 0) for token in tokens])

        return sentiments


    def tokens_with_sentiment(self, tokens, sentiment):
        """Return tokens with specified sentiment
        """
        sentiments = self.tokens_to_sentiments(tokens)
        tokens = np.array(tokens)[sentiments == sentiment]

        return tokens


if __name__ == '__main__':
    train_dataset, valid_dataset, test_dataset, warmup_dataset, vectorizer, scaler = load_data('sms', 'tfidf', 0.1,
                                                           0.1, 0., np.random.RandomState(0))
    print(len(train_dataset))
    print(len(valid_dataset))
    print(len(test_dataset))