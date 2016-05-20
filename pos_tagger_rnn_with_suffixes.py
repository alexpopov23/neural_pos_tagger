# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import random
from tensorflow.models.rnn import rnn, rnn_cell
import word2vec_optimized_utf8 as w2v
from nltk.corpus import brown
import BTBReader
from word_to_suffix import get_suffix
import pickle
import csv


''' Path to gold corpus (if set to "brown", use the nltk data) '''
gold_data = "/home/alexander/dev/projects/BAN/pos_tagger_rnn/BTB_pos_gold"
#gold_data = "brown"


''' Paths to the embeddings model '''
embeddings_save_path = "/home/alexander/dev/projects/BAN/word-embeddings/model-bg-10-05-wordforms-size200-30iters" # Bulgarian embeddings (wordforms, ~220mil)
#embeddings_save_path = "/home/alexander/dev/projects/BAN/word-embeddings/model-en/"
#embeddings_save_path = "/home/user/dev/neural-pos-tagger/word-embeddings/word-embeddings/model-en/"
embeddings_train_data = "/home/alexander/dev/projects/BAN/word-embeddings/Corpora_03.05.16/corpus_wordforms.txt"
#embeddings_train_data = "/home/alexander/dev/projects/BAN/word-embeddings/text8"
#embeddings_train_data = "/home/user/dev/neural-pos-tagger/word-embeddings/word-embeddings/text8"
embeddings_eval_data = "/home/alexander/dev/projects/BAN/word-embeddings/analogies-en.txt"
#embeddings_eval_data = "/home/user/dev/neural-pos-tagger/word-embeddings/word-embeddings/analogies-en.txt"

''' Paths to the suffix-embeddings model '''
suff_embeddings_save_path = "/home/alexander/dev/projects/BAN/suffix-embeddings/model-03.05.16_20iters"
suff_embeddings_train_data = "/home/alexander/dev/projects/BAN/suffix-embeddings/SuffixData.txt"
suff_embeddings_eval_data = "/home/alexander/dev/projects/BAN/word-embeddings/analogies-en.txt"

''' Network Parameters '''
learning_rate = 0.3 # Update rate for the weights
training_iters = 100000 # Number of training steps
batch_size = 128 # Number of sentences passed to the network in one batch
seq_width = 50 # Max sentence length (longer sentences are cut to this length)
n_hidden = 125 # Number of features/neurons in the hidden layer
#n_classes = 12 # Number of tags in the universal tagset in nltk
n_classes = 162 # Number of tags in BTB corpus
word_embedding_size = 200 # Size of the word embedding vector + the suffix embedding vector
suff_embedding_size = 50
concat_embedding_size = word_embedding_size + suff_embedding_size


''' Get the training/validation/test data '''
if gold_data == "brown":
    data = brown.tagged_sents(tagset='universal') # Get the Brown POS-tagged corpus from nltk
else:
    data, pos_tags = BTBReader.get_tagged_sentences(gold_data, True, False)
random.shuffle(data)
print len(pos_tags)
#valid_data_list = sorted(data[:5000], key=len) # Optionally, sort the sentences by length
valid_data_list = data[:3500]
#test_data_list = sorted(data[5000:10000], key=len)
test_data_list = data[3500:7000]
#train_data_list = sorted(data[10000:], key=len)
train_data_list = data[7000:]

print "Length of the full data is " + str(len(data))
print "Length of validation data is " + str(len(valid_data_list))
print "Length of test data is " + str(len(test_data_list))
print "Length of training data is " + str(len(train_data_list))
print "Some examples from the training data: " + str(train_data_list[0:10])

''' Encode the POS tags as one-hot vectors '''
pos_dict = {}
if gold_data == "brown":
    labels = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
else:
    labels = list(pos_tags)
for label in labels:
    one_hot_pos = np.zeros([n_classes], dtype=int)
    one_hot_pos[labels.index(label)] = 1
    pos_dict[label] = one_hot_pos
#print pos_dict


embeddings = {} # Dictionary to store the normalize embeddings; keys are integers from 0 to len(vocabulary)
word_to_embedding = {} # Dictionary for the mapping between word strings and corresponding integers

'''
Convert words to embeddings and shape them according to the expected dimensions
Use word2vec_optimized and load model from stored data
'''

with tf.Graph().as_default(), tf.Session() as session:
    opts = w2v.Options()
    opts.train_data = embeddings_train_data
    opts.eval_data = embeddings_eval_data
    opts.save_path = embeddings_save_path
    opts.emb_dim = word_embedding_size
    model = w2v.Word2Vec(opts, session)
    ckpt = tf.train.get_checkpoint_state(embeddings_save_path)
    if ckpt and ckpt.model_checkpoint_path:
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("No valid checkpoint to reload a model was found!")
    embeddings = session.run(model._w_in)
    word_to_embedding = model._word2id
    embeddings = tf.nn.l2_normalize(embeddings, 1).eval()

'''
Convert suffixes to embeddings and shape them according to the expected dimensions
Use word2vec_optimized and load model from stored data
'''
suff_embeddings = {}
suff_to_embedding = {}

with tf.Graph().as_default(), tf.Session() as session:
    opts = w2v.Options()
    opts.train_data = suff_embeddings_train_data
    opts.eval_data = suff_embeddings_eval_data
    opts.save_path = suff_embeddings_save_path
    opts.emb_dim = suff_embedding_size
    model = w2v.Word2Vec(opts, session)
    ckpt = tf.train.get_checkpoint_state(suff_embeddings_save_path)
    if ckpt and ckpt.model_checkpoint_path:
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("No valid checkpoint to reload a model was found!")
    suff_embeddings = session.run(model._w_in)
    suff_to_embedding = model._word2id
    suff_embeddings = tf.nn.l2_normalize(suff_embeddings, 1).eval()

''' Method to format the data to be passed into the network '''
def format_data(data_list, use_words=True):

    data = np.empty([len(data_list), seq_width, concat_embedding_size], dtype=float)
    labels = np.empty([len(data_list), seq_width, n_classes])
    seq_length = np.empty([len(data_list)], dtype=int)
    for count, sent in enumerate(data_list):
        if len(sent) > 50:
            sent = sent[:50]
        # Create a [seq_width, embedding_size]-shaped array, pad it with empty vectors when necessary
        ''' use words + suffixes or suffixes only '''
        if use_words:
            sent_padded = [np.concatenate((embeddings[word_to_embedding[word.lower().encode("utf8")]] if word.lower().encode("utf8") in word_to_embedding
                           else embeddings[word_to_embedding["UNK"]],
                                       suff_embeddings[suff_to_embedding[get_suffix(word).encode("utf8")]] if get_suffix(word).encode("utf8") in suff_to_embedding
                           else suff_embeddings[suff_to_embedding["UNK"]])) for word,_ in sent] \
                          + (seq_width-len(sent)) * [empty_embedding]
        else:
            sent_padded = [suff_embeddings[suff_to_embedding[get_suffix(word).lower().encode("utf8")]] if get_suffix(word).lower().encode("utf8") in suff_to_embedding
                           else suff_embeddings[suff_to_embedding["UNK"]] for word,_ in sent] \
                          + (seq_width-len(sent)) * [empty_embedding]
        ''' For debugging '''
        '''
        sent_padded = []
        for word,_ in sent:
            #if word.lower() in word_to_embedding:
            #    word_embeding = embeddings[word_to_embedding[word.lower()]]
            #else:
            #    word_embedding = embeddings[word_to_embedding["UNK"]]
            if get_suffix(word).lower().encode("utf8") in suff_to_embedding:
                suffix_embedding = suff_embeddings[suff_to_embedding[get_suffix(word).lower().encode("utf8")]]
            else:
                suffix_embedding = suff_embeddings[suff_to_embedding["UNK"]]
            #embedding = np.concatenate((word_embedding, suffix_embedding))
            sent_padded.append(suffix_embedding)
        sent_padded += (seq_width-len(sent)) * [empty_embedding]
        '''
        sent_array = np.asarray(sent_padded)
        data[count] = sent_array
        sent_labels = [pos_dict[label] for _,label in sent] + (seq_width-len(sent)) * [empty_pos] # Padded vector with POS
        labels[count] = sent_labels
        seq_length[count] = len(sent) # Record the length of the sentence, needed for the RNN cell
    return data, labels, seq_length

empty_embedding = concat_embedding_size * [0.0] # Empty embedding vector, used for padding
empty_pos = n_classes * [0] # Empty one-hot pos representation vector, used for padding

''' Set up validation data '''
valid_data, valid_labels, valid_seq_length = format_data(valid_data_list, True)

''' Set up test data '''
test_data, test_labels, test_seq_length = format_data(test_data_list, True)

''' Construct tensorflow graph '''
graph = tf.Graph()
with graph.as_default():

    tf_train_dataset = tf.placeholder(tf.float32, [batch_size, seq_width, concat_embedding_size])
    tf_train_labels = tf.placeholder(tf.float32, [batch_size, seq_width, n_classes])
    tf_train_seq_length = tf.placeholder(tf.int64, [batch_size])
    tf_valid_dataset = tf.constant(valid_data, tf.float32)
    tf_valid_seq_length = tf.constant(valid_seq_length, tf.int64)
    tf_test_dataset = tf.constant(test_data, tf.float32)
    tf_test_seq_length = tf.constant(test_seq_length, tf.int64)

    ''' Define weight matrices '''
    weights = {
        #'hidden': tf.Variable(tf.random_normal([embedding_size, 2*n_hidden])),
        #'hidden': tf.Variable(tf.random_normal([embedding_size, n_hidden])),
        #'hidden': tf.truncated_normal([embedding_size, n_hidden], stddev=0.1),
        # Out layer weights => 2*n_hidden because of concatenating outputs of foward + backward cells
        'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
        #'out': tf.Variable(tf.truncated_normal([2*n_hidden, n_classes], stddev=0.0883))
    }
    biases = {
        #'hidden': tf.Variable(tf.random_normal([2*n_hidden])),
        #'hidden': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    ''' Bidirectional recurrent neural network with LSTM cells '''
    def BiRNN (inputs, _seq_length):

        # input shape: (batch_size, seq_width, embedding_size) ==> (seq_width, batch_size, embedding_size)
        inputs = tf.transpose(inputs, [1, 0, 2])
        # Reshape before feeding to hidden activation layers
        inputs = tf.reshape(inputs, [-1, concat_embedding_size])
        # Hidden activation
        #inputs = tf.nn.relu(tf.matmul(inputs, weights['hidden']) + biases['hidden'])
        # Split the inputs to make a list of inputs for the rnn
        inputs = tf.split(0, seq_width, inputs) # seq_width * (batch_size, n_hidden)

        initializer = tf.random_uniform_initializer(-1,1)

        with tf.variable_scope('forward'):
            #fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
            fw_cell = rnn_cell.LSTMCell(n_hidden, concat_embedding_size, initializer=initializer)
        with tf.variable_scope('backward'):
            #bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
            bw_cell = rnn_cell.LSTMCell(n_hidden, concat_embedding_size, initializer=initializer)

        # Get lstm cell output
        outputs,_,_ = rnn.bidirectional_rnn(fw_cell, bw_cell, inputs, dtype="float32", sequence_length=_seq_length)
        outputs_tensor = tf.reshape(tf.concat(0, outputs),[-1, 2*n_hidden])

        logits = []

        for i in xrange(len(outputs)):
            final_transformed_val = tf.matmul(outputs[i],weights['out']) + biases['out']
            '''
            # TODO replace with zeroes where sentences are shorter and biases should not be calculated
            for length in tf_train_seq_length:
                tf.shape()
                if length <= i:
                    final_transformed_val[tf_train_seq_length.index(length)] = empty_pos
            '''
            logits.append(final_transformed_val)
        logits = tf.reshape(tf.concat(0, logits), [-1, n_classes])

        return logits, outputs_tensor

    with tf.variable_scope("BiRNN") as scope:
        logits, _outputs_tensor = BiRNN(tf_train_dataset, tf_train_seq_length)
        scope.reuse_variables()
        loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(
            logits, tf.reshape(tf.transpose(tf_train_labels, [1,0,2]), [-1, n_classes])))

        # try out a decaying learning rate
        #global_step = tf.Variable(0)  # count the number of steps taken.
        #learning_rate = tf.train.exponential_decay(learning_rate, global_step, 3500, 0.86, staircase=True)

        # calculate gradients, clip them and update model in separate steps
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss)
        capped_gradients = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gradients if grad!=None]
        optimizer_t = optimizer.apply_gradients(capped_gradients)
        #optimizer_t = optimizer.apply_gradients(capped_gradients)


        # calculate gradients and update model in one step
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

         # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(BiRNN(tf_valid_dataset, tf_valid_seq_length)[0])
        test_prediction = tf.nn.softmax(BiRNN(tf_test_dataset, tf_test_seq_length)[0])

''' Create a new batch from the training data (data, labels and sequence lengths) '''
def new_batch (offset):

    batch = train_data_list[offset:(offset+batch_size)]
    train_data, train_labels, seq_length = format_data(batch, True)
    return train_data, train_labels, seq_length

''' Function to calculate the accuracy on a batch of results and gold labels '''
def accuracy (predictions, labels):

    reshaped_labels = np.reshape(np.transpose(labels, (1,0,2)), (-1,n_classes))
    matching_cases = 0
    eval_cases = 0
    # Do not count results beyond the end of a sentence (in the case of sentences shorter than 50 words)
    for i in xrange(reshaped_labels.shape[0]):
        # If all values in a gold POS label are zeros, skip this calculation
        if max(reshaped_labels[i]) == 0:
            continue
        if np.argmax(reshaped_labels[i]) == np.argmax(predictions[i]):
            matching_cases+=1
        eval_cases+=1
    return (100.0 * matching_cases) / eval_cases

    '''
    # for debugging:
    argmax1 = np.argmax(predictions,1)
    reshaped_labels = np.reshape(np.transpose(labels, (1,0,2)), (-1,n_classes))
    argmax2 = np.argmax(reshaped_labels,1)
    comparison = (argmax1 == argmax2)
    sum = np.sum(comparison)
    return (100.0 * sum) / predictions.shape[0]
    #return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(np.reshape(labels, (-1, n_classes)), 1))
    #        / predictions.shape[0])
    '''

''' Run the tensorflow graph '''
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(training_iters):
        offset = (step * batch_size) % (len(train_data_list) - batch_size)
        batch_data, batch_labels, batch_seq_length = new_batch(offset)
        #batch_seq_length = batch_size * [2]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, tf_train_seq_length: batch_seq_length}
        _, l, predictions, outputs_tensor = session.run(
          [optimizer_t, loss, train_prediction, _outputs_tensor], feed_dict=feed_dict)
        if (step % 50 == 0):
          print 'Minibatch loss at step ' + str(step) + ': ' + str(l)
          print 'Minibatch accuracy: ' + str(accuracy(predictions, batch_labels))
          print 'Validation accuracy: ' + str(accuracy(
            valid_prediction.eval(), valid_labels))
    print 'Test accuracy: ' + str(accuracy(test_prediction.eval(), test_labels))