from SemEval2017.DataLoader import *
from DataUtils import text_to_wordlist, padding
import tensorflow as tf
import os
import json
from MatchLSTM.Match_LSTM import MatchLSTM
import pickle
from keras.preprocessing.text import Tokenizer
from subprocess import Popen, PIPE
import numpy as np

tf.flags.DEFINE_string("model_dir", "runs", "models where model saved")
tf.flags.DEFINE_string("Scorer", "SemEval2017/data/gold/_scorer/ev.py", "ev script")
tf.flags.DEFINE_string("GOLD_FILE",
                       "SemEval2017/data/gold/_gold/SemEval2017-Task3-CQA-QL-test.xml.subtaskB.relevancy",
                       "gold label file")
tf.flags.DEFINE_string("score_file", "score.txt", "")
tf.flags.DEFINE_string("token_file", "token_obj.pkl", "tokenizer pickle file")
FLAGS = tf.flags.FLAGS

def sent2ids(sent_dict, tokenizer):
  sent_ids_dict = {}
  for sent in sent_dict:
    sent_ids_dict[sent] = tokenizer.texts_to_sequences([text_to_wordlist(sent_dict[sent])])
  return sent_ids_dict
    
if __name__ == "__main__":
  checkpoint_dir = os.path.abspath(os.path.join(FLAGS.model_dir, "checkpoints"))
  if not (os.path.exists(checkpoint_dir) and os.path.exists(FLAGS.model_dir)):
    raise "checkpoint not exits"
  model_config = json.load(open(os.path.join(FLAGS.model_dir, "config.json")))
  print(model_config)
  with tf.Session() as sess:
    model = MatchLSTM(model_config["max_len"],
                      model_config["hidden_layer"],
                      model_config["num_class"],
                      model_config["len_vocab"],
                      model_config["embedd_dim"],
    )
    saver = tf.train.Saver()
    last_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    saver.restore(sess,last_checkpoint)
    print("loaded model from checkpoint {}".format(last_checkpoint))
    tokenizer = pickle.load(open(FLAGS.token_file, "rb"))
    test_pair_dict, test_sent_dict = get_test_set()
    sent_ids_dict = sent2ids(test_sent_dict, tokenizer)

    def gen_score(query_id, hypothesis_id, sess, model):
      query, hypothesis = sent_ids_dict[query_id], sent_ids_dict[hypothesis_id]
      feed_dict = {
        model.queries : padding(query[0], model_config["max_len"])[0],
        model.hypothesis : padding(hypothesis[0], model_config["max_len"])[0],
        model.queries_length : [len(query[0])],
        model.hypothesis_length : [len(hypothesis[0])]
        }
      score = sess.run(model.pred_label, feed_dict=feed_dict)

      label = "true" if score[0][0] >= 0.5 else "false"
      return "{}\t{}\t0\t{}\t{}\n".format(query_id, hypothesis_id, score[0][0], label)

    with open("final_score", "w") as f:
      for query_id in test_pair_dict:
        for hypothesis_id in test_pair_dict[query_id]:
          result = gen_score(query_id, hypothesis_id, sess, model)
          print(result)
          f.write(result)
    
    
  command = ['python2', FLAGS.Scorer, FLAGS.GOLD_FILE, 'final_score']
  process = Popen(command, stdout=PIPE, stdin=PIPE, stderr=PIPE)
  output, _ = process.communicate()
  print(output.decode())
  with open(os.path.join(FLAGS.model_dir, 'report.txt'), "w") as f:
    f.write(output.decode())
