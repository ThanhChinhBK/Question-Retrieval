import tensorflow as tf
import pickle
import numpy as np
import datetime
from Match_LSTM import MatchLSTM
from Rnet import Rnet
import json
import os
import DataUtils
from nltk.tokenize import word_tokenize
from tqdm import *
from sklearn.metrics import accuracy_score
import random



tf.flags.DEFINE_string("mode", "pretrained", "pretrained/tranfer")
# Training hyperparameter config
tf.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.flags.DEFINE_integer("epochs", 160, "epochs")
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
tf.flags.DEFINE_float("grad_clip", 5.0, "")
# LSTM config
tf.flags.DEFINE_integer("hidden_layer", 300, "")
tf.flags.DEFINE_integer("pad", 610, "")
tf.flags.DEFINE_float("dropout", 0.3, "")
tf.flags.DEFINE_string("Ddim", "2", "")
tf.flags.DEFINE_boolean("bidi", True, "")
tf.flags.DEFINE_string("rnnact", "tanh", "")
tf.flags.DEFINE_string("bidi_mode", "concatenate", "")
tf.flags.DEFINE_boolean("use_cudnn", True, "")
# word vector config
tf.flags.DEFINE_string(
    "embedding_path", "glove.6B.300d.txt", "word embedding path")
# Tensorflow config
tf.flags.DEFINE_integer("num_checkpoints", 5,
                        "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("out_dir", "runs/", "path to save checkpoint")
tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        if token == "``" or token=="''":
            token = '"'
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans

def load_data_from_file(dsfile):
    q, q_l = [], [] # a set of questions
    sents, s_l = [], [] # a set of sentences
    labels = [] # a set of labels
    with open(dsfile) as f:          
        for l in f:
            label = l.strip().split("\t")[2]
            qtext = l.strip().split("\t")[0]
            stext = l.strip().split("\t")[1]
            q_tok = word_tokenize(qtext.lower())
            s_tok = word_tokenize(stext.lower())
            q.append(q_tok)
            q_l.append(min(len(q_tok), FLAGS.pad))
            sents.append(s_tok)
            s_l.append(min(len(s_tok), FLAGS.pad))
            labels.append(int(label))
    return q, sents


def make_model_inputs(qi, si, q, sents, y):
    inp = {'qi': qi, 'si': si, 'q':q, 'sents':sents, 'y':y} 
    
    return inp
 
def load_set(fname, vocab=None, iseval=False):
    examples = load_data_SQUAD(fname)
    if not iseval:
        if vocab == None:
            q, sents = load_data_from_file("SemEval/train.txt")
            vocab =  DataUtils.Vocabulary(q + sents)
            update = []
            for e in examples:
                update += [e["context_tokens"], e["ques_tokens"]]
            vocab.update(update)
        else:
            update = []
            for e in examples:
                update += [e["context_tokens"], e["ques_tokens"]]
            vocab.update(update)

    
    pad = FLAGS.pad
    qis, sis, q, sents, y = [], [], [], [], []
    for e in examples:
        qi = e["ques_tokens"]  
        si = e["context_tokens"]
        qis.append(qi)
        sis.append(si)
        q.append(e["ques_tokens"])
        sents.append(e["context_tokens"])
        y.append(e["y"])
    qis = vocab.vectorize(qis, 50)
    sis = vocab.vectorize(sis, pad)
    inp = make_model_inputs(qis, sis, q, sents, y)
    if iseval:
        return (inp, y)
    else:
        return (inp, y, vocab)        
    

def load_data_SQUAD(filename):
    examples = []
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", "' ").replace("``", "' ").lower()
                
                context_tokens = word_tokenize(context)
                spans = convert_idx(context, context_tokens)

                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ').lower()
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]

                    answer = qa["answers"][0]
                    answer_text = answer["text"]
                    answer_start = answer['answer_start']
                    answer_end = answer_start + len(answer_text)
                    answer_span = []
                    for idx, span in enumerate(spans):
                        if not (answer_end <= span[0] or answer_start >= span[1]):
                            answer_span.append(idx)
                    y1, y2 = answer_span[0], answer_span[-1]
                    y = [y1,y2]
                    example = {"context_tokens": context_tokens, "ques_tokens": ques_tokens,
                               "y":y,}
                    if y2 >= 610:
                        print(context)
                        print(y2)
                        print(len(context))
                    examples.append(example)
                    
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
        return examples

def read_data(trainf, valf):
    global vocab, inp_tr, inp_val, inp_test, y_train, y_val, y_test
    
    inp_tr, y_train, vocab = load_set(trainf, iseval=False)
    
    inp_val, y_val = load_set(valf, vocab=vocab, iseval=True)

def train_step(sess, model, data_batch):
    q_batch, s_batch, y_batch = data_batch
    feed_dict = {
        model.queries : q_batch,
        #model.queries_length : ql_batch,
        model.hypothesis : s_batch,
        #model.hypothesis_length : sl_batch,
        model.dropout : FLAGS.dropout,
        model.y_SQUAD : y_batch
    }
    _, loss = sess.run([model.train_op_SQUAD, model.loss_SQUAD], feed_dict=feed_dict)
    return loss

def train_step(sess, model, data_batch):
    q_batch, s_batch, y_batch = data_batch
    feed_dict = {
        model.queries : q_batch,
        #model.queries_length : ql_batch,
        model.hypothesis : s_batch,
        #model.hypothesis_length : sl_batch,
        model.dropout : FLAGS.dropout,
        model.y_SQUAD : y_batch
    }
    _, loss = sess.run([model.train_op_SQUAD, model.loss_SQUAD], feed_dict=feed_dict)
    return loss

def test_step(sess, model, test_data):
    q_test, s_test, y_test = test_data
    final_pred = []
    final_loss = []
    for i in range(0, len(y_test), FLAGS.batch_size):
        feed_dict = {
            model.queries : q_test[i:i+FLAGS.batch_size],
            #model.queries_length : ql_test[i:i+FLAGS.batch_size],
            model.hypothesis : s_test[i:i+FLAGS.batch_size],
            #model.hypothesis_length : sl_test[i:i+FLAGS.batch_size],
            model.y_SQUAD : y_test[i:i+FLAGS.batch_size],
            model.dropout : 1.0
        }
        loss = sess.run([model.loss_SQUAD], feed_dict=feed_dict)
        final_loss.append(loss)
    print("loss in valid set :{}".format(np.mean(final_loss)))
    
    return np.mean(final_loss)

    
if __name__ == "__main__":
    trainf = os.path.join('SQUAD/train-v1.1.json')
    valf = os.path.join('SQUAD/dev-v1.1.json')
    best_map = 100
    best_epoch = 0
    print("Load data")
    read_data(trainf, valf)
    pickle.dump(vocab, open("vocab.pkl","wb"))
    print("Load Glove")
    emb = DataUtils.GloVe(FLAGS.embedding_path)
    session_conf = tf.ConfigProto(
         allow_soft_placement=FLAGS.allow_soft_placement,
         log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf) 
    model = MatchLSTM(FLAGS, vocab, emb)
    
    checkpoint_dir = os.path.abspath(os.path.join(FLAGS.out_dir, "checkpoints"))
    if not os.path.exists(checkpoint_dir):
         os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
    test_data = [ inp_val['qi'],
                  inp_val['si'],
                  y_val
    ]
    sess.run(tf.global_variables_initializer())

    for e in range(FLAGS.epochs):
        t = tqdm(range(0, len(y_train), FLAGS.batch_size), desc='train loss: %.6f' %0.0, ncols=90)
        for i in t:
            data_batch = [ inp_tr['qi'][i:i+FLAGS.batch_size],
                           inp_tr['si'][i:i+FLAGS.batch_size],
                           
                           y_train[i:i+FLAGS.batch_size]
            ]
            loss = train_step(sess, model, data_batch)
            t.set_description("epoch %d: train loss %.6f" % (e, loss))
            t.refresh()
        curr_map = test_step(sess, model, test_data)
        print("best loss in dev: %.6f" %best_map)
        if curr_map < best_map:
            best_map = curr_map
            best_epoch = e
            save_path = saver.save(sess, os.path.join(checkpoint_dir, "checkpoint"), e)
            print("saved in %s" %save_path)
