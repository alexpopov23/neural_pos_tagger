import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
from tensorflow.models.embedding import word2vec_optimized as w2v
import numpy as np
from nltk.corpus import brown
import pickle
import os

embeddings_save_path = "/home/alexander/dev/projects/BAN/word-embeddings/model-en/"
embeddings_train_data = "/home/alexander/dev/projects/BAN/word-embeddings/text8"
embeddings_eval_data = "/home/alexander/dev/projects/BAN/word-embeddings/analogies-en.txt"
pickle_folder = "/home/alexander/dev/projects/BAN/pos_tagger_rnn/Pickled"

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128

# Network Parameters
seq_width = 50 # Max sentence length
n_hidden = 128 # hidden layer num of features
n_classes = 12 # Number of tags in the universal tagset in nltk
embedding_size = 200

data = brown.tagged_sents(tagset='universal')
valid_data_list = sorted(data[:5000], key=len)
test_data_list = sorted(data[5000:10000], key=len)
train_data_list = sorted(data[10000:], key=len)
# only for dev purposes cut out a small slice of the data and use that
valid_data_list = valid_data_list[:500]
test_data_list = test_data_list[:500]
train_data_list = train_data_list[:1000]

print "Length of validation data is " + str(len(valid_data_list))
print "Length of test data is " + str(len(test_data_list))
print "Length of training data is " + str(len(train_data_list))
print train_data_list[0]
print train_data_list[-1]

''' Encode the POS tags as one-hot vectors '''
pos_dict = {}
labels = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
for label in labels:
    one_hot_pos = np.zeros([n_classes], dtype=int)
    one_hot_pos[labels.index(label)] = 1
    pos_dict[label] = one_hot_pos
print pos_dict

# convert words to embeddings and shape them according to the expected dimensions
with tf.Graph().as_default(), tf.Session() as session:
    opts = w2v.Options()
    opts.train_data = embeddings_train_data
    opts.eval_data = embeddings_eval_data
    opts.save_path = embeddings_save_path
    model = w2v.Word2Vec(opts, session)
    ckpt = tf.train.get_checkpoint_state(embeddings_save_path)
    if ckpt and ckpt.model_checkpoint_path:
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("No valid checkpoint to reload a model was found!")
    train_data = np.empty([len(train_data_list), seq_width, embedding_size], dtype=float)
    train_labels = np.empty([len(train_data_list), seq_width, n_classes])
    valid_data = np.empty([len(valid_data_list), seq_width, embedding_size], dtype=float)
    valid_labels = np.empty([len(valid_data_list), seq_width, n_classes])
    test_data = np.empty([len(test_data_list), seq_width, embedding_size], dtype=float)
    test_labels = np.empty([len(test_data_list), seq_width, n_classes])
    empty_embedding = embedding_size * [0.0]
    empty_pos = n_classes * [0]
    for sent in train_data_list:
        embeddings = session.run(model._w_in)
        sent_padded = [embeddings[model._word2id[word]] if word in model._word2id else embeddings[model._word2id["UNK"]] for word,_ in sent] \
                      + (seq_width-len(sent)) * [empty_embedding]
        sent_array = np.asarray(sent_padded)
        train_data[train_data_list.index(sent)] = sent_array
        train_labels_sent = [pos_dict[label] for _,label in sent] + (seq_width-len(sent)) * [empty_pos]
        train_labels[train_data_list.index(sent)] = train_labels_sent
    #TODO: same for test_data and valid_data, if it works
    for sent in valid_data_list:
        embeddings = session.run(model._w_in)
        sent_padded = [embeddings[model._word2id[word]] if word in model._word2id else embeddings[model._word2id["UNK"]] for word,_ in sent] \
                      + (seq_width-len(sent)) * [empty_embedding]
        sent_array = np.asarray(sent_padded)
        valid_data[valid_data_list.index(sent)] = sent_array
        valid_labels_sent = [pos_dict[label] for _,label in sent] + (seq_width-len(sent)) * [empty_pos]
        valid_labels[valid_data_list.index(sent)] = valid_labels_sent
    for sent in test_data_list:
        embeddings = session.run(model._w_in)
        sent_padded = [embeddings[model._word2id[word]] if word in model._word2id else embeddings[model._word2id["UNK"]] for word,_ in sent] \
                      + (seq_width-len(sent)) * [empty_embedding]
        sent_array = np.asarray(sent_padded)
        test_data[test_data_list.index(sent)] = sent_array
        test_labels_sent = [pos_dict[label] for _,label in sent] + (seq_width-len(sent)) * [empty_pos]
        test_labels[test_data_list.index(sent)] = test_labels_sent

pickle.dump(train_data, open(os.path.join(pickle_folder, "train_data.p"), "wb"))
pickle.dump(train_labels, open(os.path.join(pickle_folder, "train_labels.p"), "wb"))
pickle.dump(valid_data, open(os.path.join(pickle_folder, "valid_data.p"), "wb"))
pickle.dump(valid_labels, open(os.path.join(pickle_folder, "valid_labels.p"), "wb"))
pickle.dump(test_data, open(os.path.join(pickle_folder, "test_data.p"), "wb"))
pickle.dump(test_labels, open(os.path.join(pickle_folder, "test_labels.p"), "wb"))

graph = tf.Graph()

with graph.as_default():

    tf_train_dataset = tf.placeholder(tf.float32, [batch_size, seq_width, embedding_size])
    tf_train_labels = tf.placeholder(tf.float32, [batch_size, seq_width, n_classes])
    seq_length = tf.placeholder(tf.int64, [batch_size])
    #inputs = [tf.reshape(i, (batch_size, embedding_size)) for i in tf.split(1, seq_width, tf_train_dataset)]

    # Define weights
    weights = {
        # Hidden layer weights => 2*n_hidden because of foward + backward cells
        #'hidden': tf.Variable(tf.random_normal([embedding_size, 2*n_hidden])),
        'hidden': tf.Variable(tf.random_normal([embedding_size, n_hidden])),
        'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
    }
    biases = {
        #'hidden': tf.Variable(tf.random_normal([2*n_hidden])),
        'hidden': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    def BiRNN (inputs, _weights, _biases, _seq_length):

        # input shape: (batch_size, seq_width, embedding_size)
        inputs = tf.transpose(inputs, [1, 0, 2])
        # Reshape before feeding to hidden activation layers
        inputs = tf.reshape(inputs, [-1, embedding_size])
        # Hidden activation
        inputs = tf.sigmoid(tf.matmul(inputs, weights['hidden']) + biases['hidden'])
        # Split the inputs to make a list of inputs for the rnn
        inputs = tf.split(0, seq_width, inputs) # seq_width * (batch_size, 2*n_hidden)

        initializer = tf.random_uniform_initializer(-1,1)

        fw_cell = rnn_cell.LSTMCell(n_hidden, n_hidden, initializer=initializer)
        bw_cell = rnn_cell.LSTMCell(n_hidden, n_hidden, initializer=initializer)

        # Get lstm cell output
        outputs = rnn.bidirectional_rnn(fw_cell, bw_cell, inputs, dtype="float32", sequence_length=_seq_length)

        logits = []

        for i in xrange(len(outputs)):
            final_transformed_val = tf.matmul(outputs[i],weights['out']) + biases['out']
            logits.append(tf.nn.softmax(final_transformed_val))
        logits = tf.reshape(tf.concat(1, logits), [-1, n_classes])

        return logits

    logits = BiRNN(tf_train_dataset, weights, biases, seq_length)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
        logits, tf.reshape(tf_train_labels, [-1, n_classes])))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

     # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    #valid_prediction = tf.nn.softmax(BiRNN(tf_valid_dataset, weights, biases, seq_length))
    #test_prediction = tf.nn.softmax(BiRNN(tf_test_dataset, weights, biases, seq_length))
'''
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    #TODO
    # take batch_size of sentences from train_data, lookup embeddings for each word and construct tensor
    # also, padd every sentence to seq_width with zero vectors
    #TODO
    # feed into graph and run session
'''