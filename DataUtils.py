import re
import pickle
import numpy as np
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
import numpy as np
import json
from operator import itemgetter

from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences

import re
from collections import namedtuple

from keras.callbacks import Callback


def remove_html_tag(text):
    return re.sub(r'\[\S+[|\]]([\S ]+){0,1}\]', '. ', text)


def remove_url(text):
    return re.sub(r'[(http|ftp|https)://]*([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',
                  'urllink', text)


def remove_emo(text):
    text = re.sub(r'(?::|;|=)(?:-)?(?:\)|\(|D|P|S|o|O)+', 'emoji', text)
    return text


def clear_short_sent_text(text):
    text_sents = sent_tokenize(text)
    text_sents = [word_tokenize(
        t)[:-1] for t in text_sents[:-1]] + [word_tokenize(text_sents[-1])]
    return '. '.join([' '.join(t) for t in text_sents if len(t) > 0])


def extract_acronym(text):
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"ive", "i have", text)
    text = re.sub(r"iam", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'r", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    return text


def removePunctuation(question):
    return re.sub('[^\w\s]+', ' . ', question)


def clean_text(text, remove_stopwords=False, stem_words=False):
    # Convert words to lower case and split them
    # xt = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    # text = " ".join(text)
    text = remove_html_tag(text)
    text = remove_url(text)
    text = remove_emo(text)
    text = extract_acronym(text)
    text = removePunctuation(text)
    text = clear_short_sent_text(text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    return text.lower()


class Embedder(object):
    """ Generic embedding interface.

    Required: attributes g and N """

    def map_tokens(self, tokens, ndim=2):
        """ for the given list of tokens, return a list of GloVe embeddings,
        or a single plain bag-of-words average embedding if ndim=1.

        Unseen words (that's actually *very* rare) are mapped to 0-vectors. """
        gtokens = [self.g[t] for t in tokens if t in self.g]
        if not gtokens:
            return np.zeros((1, self.N)) if ndim == 2 else np.zeros(self.N)
        gtokens = np.array(gtokens)
        if ndim == 2:
            return gtokens
        else:
            return gtokens.mean(axis=0)

    def map_set(self, ss, ndim=2):
        """ apply map_tokens on a whole set of sentences """
        return [self.map_tokens(s, ndim=ndim) for s in ss]

    def pad_set(self, ss, spad, N=None):
        """ Given a set of sentences transformed to per-word embeddings
        (using glove.map_set()), convert them to a 3D matrix with fixed
        sentence sizes - padded or trimmed to spad embeddings per sentence.

        Output is a tensor of shape (len(ss), spad, N).

        To determine spad, use something like
            np.sort([np.shape(s) for s in s0], axis=0)[-1000]
        so that typically everything fits, but you don't go to absurd lengths
        to accomodate outliers.
        """
        ss2 = []
        if N is None:
            N = self.N
        for s in ss:
            if spad > s.shape[0]:
                if s.ndim == 2:
                    s = np.vstack((s, np.zeros((spad - s.shape[0], N))))
                else:  # pad non-embeddings (e.g. toklabels) too
                    s = np.hstack((s, np.zeros(spad - s.shape[0])))
            elif spad < s.shape[0]:
                s = s[:spad]
            ss2.append(s)
        return np.array(ss2)


class GloVe(Embedder):
    """ A GloVe dictionary and the associated N-dimensional vector space """

    def __init__(self, glovepath):
        """ Load GloVe dictionary from the standard distributed text file.

        Glovepath should contain %d, which is substituted for the embedding
        dimension N. """
        self.N = None
        self.g = dict()
        self.glovepath = glovepath 

        with open(self.glovepath, 'r') as f:
            for line in f:
                l = line.split()
                word = l[0]
                self.g[word] = np.array(l[1:]).astype(float)
                if self.N == None:
                    self.N = len(l) - 1
                    print("embedding dim is setted to {}".format(self.N))

def hash_params(pardict):
    ps = json.dumps(dict([(k, str(v))
                          for k, v in pardict.items()]), sort_keys=True)
    h = hash(ps)
    return ps, h


"""
NLP preprocessing tools for sentences.

Currently, this just tags the token sequences by some trivial boolean flags
that denote some token characteristics and sentence-sentence overlaps.

In principle, this module could however include a lot more sophisticated
NLP tagging pipelines, or loading precomputed such data.
"""

stop = stopwords.words('english')

flagsdim = 4


def sentence_flags(s0, s1, spad):
    """ For sentence lists s0, s1, generate numpy tensor
    (#sents, spad, flagsdim) that contains a sparse indicator vector of
    various token properties.  It is meant to be concatenated to the token
    embedding. """

    def gen_iflags(s, spad):
        iflags = []
        for i in range(len(s)):
            iiflags = [[False, False] for j in range(spad)]
            for j, t in enumerate(s[i]):
                if j >= spad:
                    break
                number = False
                capital = False
                if re.match('^[0-9\W]*[0-9]+[0-9\W]*$', t):
                    number = True
                if j > 0 and re.match('^[A-Z]', t):
                    capital = True
                iiflags[j] = [number, capital]
            iflags.append(iiflags)
        return iflags

    def gen_mflags(s0, s1, spad):
        """ generate flags for s0 that represent overlaps with s1 """
        mflags = []
        for i in range(len(s0)):
            mmflags = [[False, False] for j in range(spad)]
            for j in range(min(spad, len(s0[i]))):
                unigram = False
                bigram = False
                for k in range(len(s1[i])):
                    if s0[i][j].lower() != s1[i][k].lower():
                        continue
                    # do not generate trivial overlap flags, but accept them as
                    # part of bigrams
                    if s0[i][j].lower() not in stop and not re.match('^\W+$', s0[i][j]):
                        unigram = True
                    try:
                        if s0[i][j + 1].lower() == s1[i][k + 1].lower():
                            bigram = True
                    except IndexError:
                        pass
                mmflags[j] = [unigram, bigram]
            mflags.append(mmflags)
        return mflags

    # individual flags (for understanding)
    iflags0 = gen_iflags(s0, spad)
    iflags1 = gen_iflags(s1, spad)

    # s1-s0 match flags (for attention)
    mflags0 = gen_mflags(s0, s1, spad)
    mflags1 = gen_mflags(s1, s0, spad)

    return [np.dstack((iflags0, mflags0)),
            np.dstack((iflags1, mflags1))]


"""
Vocabulary that indexes words, can handle OOV words and integrates word
embeddings.
"""
class Vocabulary:
    """ word-to-index mapping, token sequence mapping tools and
    embedding matrix construction tools """

    def __init__(self, sentences, count_thres=1):
        """ build a vocabulary from given list of sentences, but including
        only words occuring at least #count_thres times """

        # Counter() is superslow :(
        vocabset = defaultdict(int)
        for s in sentences:
            for t in s:
                vocabset[t] += 1

        vocab = sorted(list(map(itemgetter(0),
                                filter(lambda k: itemgetter(1)(k) >= count_thres,
                                       vocabset.items()))))
        self.word_idx = dict((w, i + 2) for i, w in enumerate(vocab))
        self.word_idx['_PAD_'] = 0
        self.word_idx['_OOV_'] = 1
        print('Vocabulary of %d words' % (len(self.word_idx)))

        self.embcache = dict()

    def update(self, sentences):
        for s in sentences:
            for t in s:
                self.add_word(t)
        print('Vocabulary of %d words' % (len(self.word_idx)))

    def add_word(self, word):
        if word not in self.word_idx:
            self.word_idx[word] = len(self.word_idx)

    def vectorize(self, slist, pad=60):
        """ build an pad-ed matrix of word indices from a list of
        token sequences """
        silist = [[self.word_idx.get(t, 1) for t in s] for s in slist]
        if pad is not None:
            return pad_sequences(silist, maxlen=pad, truncating='post', padding='post')
        else:
            return silist

    def embmatrix(self, emb):
        """ generate index-based embedding matrix from embedding class emb
        (typically GloVe); pass as weights= argument of Keras' Embedding layer """
        if str(emb) in self.embcache:
            return self.embcache[str(emb)]
        embedding_weights = np.zeros((len(self.word_idx), emb.N))
        for word, index in self.word_idx.items():
            try:
                embedding_weights[index, :] = emb.g[word]
            except KeyError:
                if index == 0:
                    embedding_weights[index, :] = np.zeros(emb.N)
                else:
                    # 0.25 is embedding SD
                    embedding_weights[
                        index, :] = np.random.uniform(-0.25, 0.25, emb.N)
        self.embcache[str(emb)] = embedding_weights
        return embedding_weights

    def size(self):
        return len(self.word_idx)

class CharVocabulary:
    """ word-to-index mapping, token sequence mapping tools and
    embedding matrix construction tools """

    def __init__(self, sentences, count_thres=1):
        """ build a vocabulary from given list of sentences, but including
        only words occuring at least #count_thres times """

        # Counter() is superslow :(
        vocabset = defaultdict(int)
        for s in sentences:
            for t in s:
                for z in t:
                    vocabset[z] += 1

        vocab = sorted(list(map(itemgetter(0),
                                filter(lambda k: itemgetter(1)(k) >= count_thres,
                                       vocabset.items()))))
        self.char_idx = dict((w, i + 2) for i, w in enumerate(vocab))
        self.char_idx['_PAD_'] = 0
        self.char_idx['_OOV_'] = 1
        print('Char Vocabulary of %d words' % (len(self.char_idx)))

        self.embcache = dict()

    def update(self, sentences):
        for s in sentences:
            for t in s:
                for z in t:
                    self.add_word(z)
        print('Char Vocabulary of %d words' % (len(self.char_idx)))

    def add_word(self, word):
        if word not in self.char_idx:
            self.char_idx[word] = len(self.char_idx)

    def vectorize(self, slist, pad=15, seq_pad=60):
        """ build an pad-ed matrix of word indices from a list of
        token sequences """
        silist = []
        for s in slist:
            cilist = [[self.char_idx.get(t, 1) for t in c] for c in s]
            cilist = pad_sequences(cilist, maxlen=pad, truncating='post', padding='post')
            if seq_pad <= len(cilist):
                silist.append(cilist[:seq_pad])
            else:
                cilist = np.concatenate((cilist, np.zeros([seq_pad-len(cilist), pad], dtype=np.int32)), 0)
                silist.append(cilist)
        return silist

    def size(self):
        return len(self.char_idx)


"""
Evaluation tools, mainly non-straightforward methods.
"""


def aggregate_s0(s0, y, ypred, k=None):
    """
    Generate tuples (s0, [(y, ypred), ...]) where the list is sorted
    by the ypred score.  This is useful for a variety of list-based
    measures in the "anssel"-type tasks.
    """
    ybys0 = dict()
    for i in range(len(s0)):
        try:
            s0is = s0[i].tostring()
        except AttributeError:
            s0is = str(s0[i])
        if s0is in ybys0:
            ybys0[s0is].append((y[i], ypred[i]))
        else:
            ybys0[s0is] = [(y[i], ypred[i])]

    for s, yl in ybys0.items():
        if k is not None:
            yl = yl[:k]
        ys = sorted(yl, key=lambda yy: yy[1], reverse=True)
        yield (s, ys)


def mrr(s0, y, ypred):
    """
    Compute MRR (mean reciprocial rank) of y-predictions, by grouping
    y-predictions for the same s0 together.  This metric is relevant
    e.g. for the "answer sentence selection" task where we want to
    identify and take top N most relevant sentences.
    """
    rr = []
    for s, ys in aggregate_s0(s0, y, ypred):
        if np.sum([yy[0] for yy in ys]) == 0:
            continue  # do not include s0 with no right answers in MRR
        ysd = dict()
        for yy in ys:
            if yy[1][0] in ysd:
                ysd[yy[1][0]].append(yy[0])
            else:
                ysd[yy[1][0]] = [yy[0]]
        rank = 0
        for yp in sorted(ysd.keys(), reverse=True):
            if np.sum(ysd[yp]) > 0:
                rankofs = 1 - np.sum(ysd[yp]) / len(ysd[yp])
                rank += len(ysd[yp]) * rankofs
                break
            rank += len(ysd[yp])
        rr.append(1 / float(1 + rank))
    return np.mean(rr)


def map_(s0, y, ypred, debug=False):
    MAP = []
    for s, ys in aggregate_s0(s0, y, ypred):
        candidates = ys
        avg_prec = 0
        precisions = []
        num_correct = 0
        for i in range(len(candidates)):
            if candidates[i][0] == 1:
                num_correct += 1
                precisions.append(num_correct / (i + 1))
        if len(precisions):
            avg_prec = sum(precisions) / len(precisions)
        MAP.append(avg_prec)
    if debug:
        return MAP
    else:
        return np.mean(MAP)

AnsSelRes = namedtuple('AnsSelRes', ['MRR', 'MAP'])


def eval_QA(pred, q, y, MAP=False):
    mrr_ = mrr(q, y, pred)
    map__ = map_(q, y, pred)
    print('MRR: %f; MAP: %f' % (mrr_, map__))

    return AnsSelRes(mrr_, map__)


"""
Task-specific callbacks for the fit() function.
"""


class AnsSelCB():
    """ A callback that monitors answer selection validation ACC after each epoch """

    def __init__(self, val_q, val_s, y, inputs):
        self.val_q = val_q
        self.val_s = val_s
        self.val_y = y
        self.val_inputs = inputs

    def on_epoch_end(self, pred):
        logs={}
        mrr_ = mrr(self.val_q, self.val_y, pred)
        map__ = map_(self.val_q, self.val_y, pred)
        print('val MRR %f; MAP: %f' % (mrr_, map__))
        logs['mrr'] = mrr_
        logs['map'] = map__
        return logs

    def on_debug(self, pred, thresold=0.6, fn="debug.txt"):
        fw = open(fn , "w")
        q_uniq = set(self.val_q)
        MAP =  map_(self.val_q, self.val_y, pred, debug=True)
        for i, _map in enumerate(MAP):
            if _map < thresold:
                fw.write("%.6f\t%s\n" %(_map, q_uniq[i]))
                for j in range(10):
                    ind = i * 10 + j
                    fw.write("%s\t%d\t%.6f\n" %(self.val_s[ind], self.val_y[ind], pred[ind]))
        fw.close()
