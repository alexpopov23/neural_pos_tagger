import tensorflow as tf
import numpy as np
import word2vec_optimized_utf8 as w2v
import BTBReader
import argparse
from tensorflow.models.rnn import rnn, rnn_cell
from nltk.corpus import brown
from word_to_suffix import get_suffix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(version='1.0',description='Train a neural POS tagger.')
    parser.add_argument('-use_word_embeddings', dest='use_word_embeddings', required=True,
                        help='State whether word embeddings should be used as input.')
    parser.add_argument('-word_embeddings_model', dest='word_embeddings_save_path', required=False,
                        help='The path to the pretrained model with the word embeddings.')
    parser.add_argument('-word_embedding_size', dest='word_embedding_size', required=False,
                        help='Size of the word embedding vectors.')
    parser.add_argument('-word_embeddings_train_data', dest='word_embeddings_train_data', required=False,
                        help='The path to the corpus used for training the word embeddings.')
    parser.add_argument('-word_embeddings_eval_data', dest='word_embeddings_eval_data', required=False,
                        help='The path to the set of analogies used for evaluation of the word embeddings.')
    parser.add_argument('-use_suffix_embeddings', dest='use_suffix_embeddings', required=True,
                        help='State whether suffix embeddings should be used as input.')
    parser.add_argument('-suffix_embeddings_model', dest='suffix_embeddings_save_path', required=False,
                        help='The path to the pretrained model with the suffix embeddings.')
    parser.add_argument('-suffix_embedding_size', dest='suffix_embedding_size', required=False,
                        help='Size of the suffix embedding vectors.')
    parser.add_argument('-suffix_embeddings_train_data', dest='suffix_embeddings_train_data', required=False,
                        help='The path to the corpus used for training the suffix embeddings.')
    parser.add_argument('-suffix_embeddings_eval_data', dest='suffix_embeddings_eval_data', required=False,
                        help='The path to the set of analogies used for evaluation of the suffix embeddings.')
    parser.add_argument('-learning_rate', dest='learning_rate', required=False, default=0.3,
                        help='How fast should the network learn.')
    parser.add_argument('-training_iterations', dest='training_iters', required=False, default=10000,
                        help='How many iterations should the network train for.')
    parser.add_argument('-batch_size', dest='batch_size', required=False, default=128,
                        help='Size of the training batches.')
    parser.add_argument('-n_hidden', dest='n_hidden', required=False, default=100,
                        help='Size of the hidden layer.')
    parser.add_argument('-sequence_width', dest='seq_width', required=False, default=50,
                        help='Maximum length of a sentence to be passed to the network (the rest is cut off).')
    parser.add_argument('-tagset_size', dest='n_classes', required=True,
                        help='Size of the POS tags in the corpus used for training/testing.')
    parser.add_argument('-gold_data', dest='gold_data', required=True, default="brown",
                        help='The path to the gold corpus used for training/testing.')

    args = parser.parse_args()

    ''' Path to gold corpus (if set to "brown", use the nltk data) '''
    gold_data = args.gold_data

    ''' Parameters of the word embedding model '''
    if args.use_word_embeddings:
        word_embeddings_save_path = args.word_embeddings_save_path
        word_embeddings_train_data = args.word_embeddings_train_data
        #word_embeddings_eval_data = args.word_embeddings_eval_data
        word_embedding_size = int(args.word_embedding_size) # Size of the word embedding vectors
    else:
        word_embedding_size = 0

    ''' Parameters of the suffix embedding model '''
    if args.use_suffix_embeddings:
        suffix_embeddings_save_path = args.suffix_embeddings_save_path
        suffix_embeddings_train_data = args.suffix_embeddings_train_data
        #suffix_embeddings_eval_data = args.suffix_embeddings_eval_data
        suffix_embedding_size = int(args.suffix_embedding_size) # Size of the suffix embedding vectors
    else:
        suffix_embedding_size = 0

    ''' Network Parameters '''
    learning_rate = float(args.learning_rate) # Update rate for the weights
    training_iters = int(args.training_iters) # Number of training steps
    batch_size = int(args.batch_size) # Number of sentences passed to the network in one batch
    seq_width = int(args.seq_width) # Max sentence length (longer sentences are cut to this length)
    n_hidden = int(args.n_hidden) # Number of features/neurons in the hidden layer
    n_classes = int(args.n_classes) # Number of tags in the gold corpus
    embedding_size = word_embedding_size + suffix_embedding_size
    if embedding_size == 0:
        print "No embedding model given as parameter!"
        exit(1)

    ''' Get the training/validation/test data '''
    if gold_data == "brown":
        data = brown.tagged_sents(tagset='universal') # Get the Brown POS-tagged corpus from nltk
    else:
        data, pos_tags = BTBReader.get_tagged_sentences(gold_data, False, False)
    #valid_data_list = sorted(data[:5000], key=len) # Optionally, sort the sentences by length
    valid_data_list = data[:1000]
    #test_data_list = sorted(data[5000:10000], key=len)
    test_data_list = data[1000:2000]
    #train_data_list = sorted(data[10000:], key=len)
    train_data_list = data[2000:]

    print "POS tags " + str(pos_tags)
    print "Number of labels in the tagset is " + str(len(pos_tags))
    print "Length of the full data is " + str(len(data))
    print "Length of validation data is " + str(len(valid_data_list))
    print "Length of test data is " + str(len(test_data_list))
    print "Length of training data is " + str(len(train_data_list))
    #print "Some examples from the training data: " + str(train_data_list[0:10])

    ''' Encode the POS tags as one-hot vectors '''
    pos_dict = {}
    outside_ner = 0
    if gold_data == "brown":
        labels = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
    else:
        labels = list(pos_tags)
    for label in labels:
        if label == "O":
            outside_ner = labels.index(label)
        one_hot_pos = np.zeros([n_classes], dtype=int)
        one_hot_pos[labels.index(label)] = 1
        pos_dict[label] = one_hot_pos

    if args.use_word_embeddings:
        embeddings = {} # Dictionary to store the normalize embeddings; keys are integers from 0 to len(vocabulary)
        word_to_embedding = {} # Dictionary for the mapping between word strings and corresponding integers

        '''
        Convert words to embeddings and shape them according to the expected dimensions
        Use word2vec_optimized and load model from stored data
        '''
        with tf.Graph().as_default(), tf.Session() as session:
            opts = w2v.Options()
            opts.train_data = word_embeddings_train_data
            #opts.eval_data = word_embeddings_eval_data
            opts.save_path = word_embeddings_save_path
            opts.emb_dim = word_embedding_size
            model = w2v.Word2Vec(opts, session)
            ckpt = tf.train.get_checkpoint_state(word_embeddings_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                model.saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("No valid checkpoint to reload a model was found!")
            embeddings = session.run(model._w_in)
            word_to_embedding = model._word2id
            embeddings = tf.nn.l2_normalize(embeddings, 1).eval()
    if args.use_suffix_embeddings:
        suff_embeddings = {}
        suff_to_embedding = {}

        '''
        Convert suffixes to embeddings and shape them according to the expected dimensions
        Use word2vec_optimized and load model from stored data
        '''

        with tf.Graph().as_default(), tf.Session() as session:
            opts = w2v.Options()
            opts.train_data = suffix_embeddings_train_data
            #opts.eval_data = suffix_embeddings_eval_data
            opts.save_path = suffix_embeddings_save_path
            opts.emb_dim = suffix_embedding_size
            model = w2v.Word2Vec(opts, session)
            ckpt = tf.train.get_checkpoint_state(suffix_embeddings_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                model.saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("No valid checkpoint to reload a model was found!")
            suff_embeddings = session.run(model._w_in)
            suff_to_embedding = model._word2id
            suff_embeddings = tf.nn.l2_normalize(suff_embeddings, 1).eval()

    ''' Method to format the data to be passed into the network '''
    def format_data(data_list):

        data = np.empty([len(data_list), seq_width, embedding_size], dtype=float)
        labels = np.empty([len(data_list), seq_width, n_classes])
        seq_length = np.empty([len(data_list)], dtype=int)
        for count, sent in enumerate(data_list):
            if len(sent) > 50:
                sent = sent[:50]
            ''' Create a [seq_width, embedding_size]-shaped array, pad it with empty vectors when necessary '''
            ''' construct the sentence representation depending on the selected embedding model '''
            if args.use_word_embeddings and args.use_suffix_embeddings:
                sent_padded = [np.concatenate((embeddings[word_to_embedding[word.lower().encode("utf8")]] if word.lower().encode("utf8") in word_to_embedding
                               else embeddings[word_to_embedding["UNK"]],
                                           suff_embeddings[suff_to_embedding[get_suffix(word).encode("utf8")]] if get_suffix(word).encode("utf8") in suff_to_embedding
                               else suff_embeddings[suff_to_embedding["UNK"]])) for word,_ in sent] \
                              + (seq_width-len(sent)) * [empty_embedding]
            elif args.use_word_embeddings:
                sent_padded = [embeddings[word_to_embedding[word.lower().encode("utf8")]] if word.lower().encode("utf8") in word_to_embedding
                               else embeddings[word_to_embedding["UNK"]] for word,_ in sent] \
                              + (seq_width-len(sent)) * [empty_embedding]
            elif args.use_suffix_embeddings:
                sent_padded = [suff_embeddings[suff_to_embedding[get_suffix(word).lower().encode("utf8")]] if get_suffix(word).lower().encode("utf8") in suff_to_embedding
                               else suff_embeddings[suff_to_embedding["UNK"]] for word,_ in sent] \
                              + (seq_width-len(sent)) * [empty_embedding]
            sent_array = np.asarray(sent_padded)
            data[count] = sent_array
            sent_labels = [pos_dict[label] for _,label in sent] + (seq_width-len(sent)) * [empty_pos] # Padded vector with POS
            labels[count] = sent_labels
            seq_length[count] = len(sent) # Record the length of the sentence, needed for the RNN cell
        return data, labels, seq_length

    empty_embedding = embedding_size * [0.0] # Empty embedding vector, used for padding
    empty_pos = n_classes * [0] # Empty one-hot pos representation vector, used for padding

    ''' Set up validation data '''
    valid_data, valid_labels, valid_seq_length = format_data(valid_data_list)

    ''' Set up test data '''
    test_data, test_labels, test_seq_length = format_data(test_data_list)

    ''' Construct tensorflow graph '''
    graph = tf.Graph()
    with graph.as_default():

        tf_train_dataset = tf.placeholder(tf.float32, [batch_size, seq_width, embedding_size])
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
            inputs = tf.reshape(inputs, [-1, embedding_size])
            # Hidden activation
            #inputs = tf.nn.relu(tf.matmul(inputs, weights['hidden']) + biases['hidden'])
            # Split the inputs to make a list of inputs for the rnn
            inputs = tf.split(0, seq_width, inputs) # seq_width * (batch_size, n_hidden)

            initializer = tf.random_uniform_initializer(-1,1)

            with tf.variable_scope('forward'):
                #fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
                #lstm1 = rnn_cell.LSTMCell(n_hidden, embedding_size, initializer=initializer)
                #lstm2 = rnn_cell.LSTMCell(n_hidden, embedding_size, initializer=initializer)
                #fw_cell = rnn_cell.MultiRNNCell([lstm1, lstm2])
                fw_cell = rnn_cell.LSTMCell(n_hidden, embedding_size, initializer=initializer)
            with tf.variable_scope('backward'):
                #bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
                #lstm3 = rnn_cell.LSTMCell(n_hidden, embedding_size, initializer=initializer)
                #lstm4 = rnn_cell.LSTMCell(n_hidden, embedding_size, initializer=initializer)
                #bw_cell = rnn_cell.MultiRNNCell([lstm3, lstm4])
                bw_cell = rnn_cell.LSTMCell(n_hidden, embedding_size, initializer=initializer)

            # Get lstm cell output
            outputs,_,_ = rnn.bidirectional_rnn(fw_cell, bw_cell, inputs, dtype="float32", sequence_length=_seq_length)
            outputs_tensor = tf.reshape(tf.concat(0, outputs),[-1, 2*n_hidden])

            logits = []

            for i in xrange(len(outputs)):
                final_transformed_val = tf.matmul(outputs[i],weights['out']) + biases['out']
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

             # Predictions for the training, validation, and test data.
            train_prediction = tf.nn.softmax(logits)
            valid_prediction = tf.nn.softmax(BiRNN(tf_valid_dataset, tf_valid_seq_length)[0])
            test_prediction = tf.nn.softmax(BiRNN(tf_test_dataset, tf_test_seq_length)[0])

    ''' Create a new batch from the training data (data, labels and sequence lengths) '''
    def new_batch (offset):

        batch = train_data_list[offset:(offset+batch_size)]
        train_data, train_labels, seq_length = format_data(batch)
        return train_data, train_labels, seq_length

    ''' Function to calculate the accuracy on a batch of results and gold labels '''
    def accuracy (predictions, labels):

        '''
        Precision = True Positives / (True Positives + False Positives)
        Recall = True Positives / (True Positives + False Negatives)
        F-1 = 2 * (Precision * Recall) / (Precision + Recall)
        '''
        reshaped_labels = np.reshape(np.transpose(labels, (1,0,2)), (-1,n_classes))
        matching_cases = 0
        eval_cases = 0
        false_positives = 0.0
        false_negatives = 0.0
        true_positives = 0.0
        # Do not count results beyond the end of a sentence (in the case of sentences shorter than 50 words)
        for i in xrange(reshaped_labels.shape[0]):
            # If all values in a gold POS label are zeros, skip this calculation
            if max(reshaped_labels[i]) == 0:
                continue
            # If label is "O", disregard true positives but count false positives
            if reshaped_labels[i][outside_ner] == 1:
                if max(predictions[i]) != outside_ner:
                    false_positives+=1
                    continue
            if np.argmax(reshaped_labels[i]) == np.argmax(predictions[i]):
                true_positives+=1
            else:
                false_negatives+=1
            eval_cases+=1
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        if precision == 0 and recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1

    ''' Run the tensorflow graph '''
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(training_iters):
            offset = (step * batch_size) % (len(train_data_list) - batch_size)
            batch_data, batch_labels, batch_seq_length = new_batch(offset)
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, tf_train_seq_length: batch_seq_length}
            _, l, predictions, outputs_tensor = session.run(
              [optimizer_t, loss, train_prediction, _outputs_tensor], feed_dict=feed_dict)
            if (step % 50 == 0):
              print 'Minibatch loss at step ' + str(step) + ': ' + str(l)
              min_prec, min_rec, min_f1 = accuracy(predictions, batch_labels)
              print 'Minibatch precision: ' + str(min_prec)[:4] + '; minibatch recall: ' + str(min_rec)[:4]\
                    + '; minibatch f-1: ' + str(min_f1)[:4]
              val_prec, val_rec, val_f1 = accuracy(valid_prediction.eval(), valid_labels)
              print 'Validation precision: ' + str(val_prec)[:4] + '; validation recall: ' + str(val_rec)[:4]\
                    + '; validation f-1: ' + str(val_f1)[:4]
        test_prec, test_rec, test_f1 = accuracy(test_prediction.eval(), test_labels)
        print 'Test precision: ' + str(test_prec)[:4] + '; test recall: ' + str(test_rec)[:4] \
              + '; test f-1: ' + str(test_f1)[:4]