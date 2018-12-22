import numpy as np
from gensim.models.word2vec import Word2Vec

def set2vec(padded_set, num_label):
  queries = np.concatenate([x[0] for x in padded_set])
  queries_length = np.asarray([x[1] for x in padded_set])
  hypothesis = np.concatenate([x[2] for x in padded_set])
  hypothesis_length = np.asarray([x[3] for x in padded_set])
  labels = np.array([x[4] for x in padded_set])
  labels_mat = np.zeros((labels.shape[0], num_label))
  labels_mat[np.arange(labels.shape[0]), labels] = 1
  return queries, queries_length, hypothesis, hypothesis_length, labels_mat

def batch(queries, queries_length, hypothesis,
          hypothesis_length, labels, batch_length, set_random=True):
  if set_random == True:
    rand_ids = np.arange(queries.shape[0])
    np.random.shuffle(rand_ids)
    queries = queries[rand_ids]
    queries_length = queries_length[rand_ids]
    hypothesis = hypothesis[rand_ids]
    hypothesis_length = hypothesis_length[rand_ids]
    labels = labels[rand_ids]
    
  data_length = len(queries)
  num_batch = data_length // batch_length
  for i in range(num_batch):
    pre = i * batch_length
    nex = (i+1) * batch_length
    yield (queries[pre:nex], queries_length[pre:nex], hypothesis[pre:nex]\
           , hypothesis_length[pre:nex], labels[pre:nex])
  yield (queries[nex:], queries_length[nex:], hypothesis[nex:]\
         , hypothesis_length[nex:], labels[nex:])

def load_wordvector(embedding, embedding_path, vocab):
  if embedding == 'word2vec':
    # loading word2vec
    embedd_dict = Word2Vec.load_word2vec_format(embedding_path, binary=True)
    embedded_dim = embedd_dict.vector_size
  elif embedding == 'glove':
    # loading GloVe
    embedd_dict = dict()
    with open(embedding_path, 'r') as file:
      for line in file:
        line = line.strip()
        if len(line) == 0:
          continue

        tokens = line.split()
        embedd_dim = len(tokens) - 1 #BECAUSE THE ZEROTH INDEX IS OCCUPIED BY THE WORD
        embedd = np.empty([1, embedd_dim], dtype=np.float64)
        embedd[:] = tokens[1:]
        embedd_dict[tokens[0]] = embedd

  embedded_matrix = np.zeros((len(vocab), embedd_dim))
  count = 0
  for word in vocab:
    if word in embedd_dict:
      count += 1
      embedded_matrix[vocab[word]] = embedd_dict[word]
    else:
      embedded_matrix[vocab[word]] = np.random.uniform(-0.25, 0.25, embedd_dim)
  print("found {} out of {} in embeded path".format(count, len(vocab)))
  return embedded_matrix, embedd_dim
