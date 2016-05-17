import numpy as np

def get_data(data_list):

    valid_data = np.empty([len(data_list), seq_width, embedding_size], dtype=float)
    valid_labels = np.empty([len(data_list), seq_width, n_classes])
    valid_seq_length = np.empty([len(data_list)], dtype=int)
    for count, sent in enumerate(data_list):
        if len(sent) > 50:
            sent = sent[:50]
        # Create a [seq_width, embedding_size]-shaped array, pad it with empty vectors when necessary
        sent_padded = [embeddings[word_to_embedding[word.lower().encode("utf8")]] if word.lower().encode("utf8") in word_to_embedding
                       else embeddings[word_to_embedding["UNK"]] for word,_ in sent] \
                      + (seq_width-len(sent)) * [empty_embedding]
        sent_array = np.asarray(sent_padded)
        valid_data[count] = sent_array
        sent_labels = [pos_dict[label] for _,label in sent] + (seq_width-len(sent)) * [empty_pos] # Padded vector with POS
        valid_labels[count] = sent_labels
        valid_seq_length[count] = len(sent) # Record the length of the sentence, needed for the RNN cell