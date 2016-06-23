import os
import numpy as np
from constants import VECTOR_POSITIONS

#vector_positions = 76

def get_morpho_dict(path_to_dict):

    morpho_dict = {}
    files = os.listdir(path_to_dict)
    for f_path in files:
        f = open(os.path.join(path_to_dict, f_path), "r")
        lines = f.readlines()
        for line in lines:
            wf, tag, lemma = line.split("\t")
            if wf not in morpho_dict:
                lemma_tag_list = []
                lemma_tag = (lemma.strip(), tag.strip())
                lemma_tag_list.append(lemma_tag)
                morpho_dict[wf.strip()] = lemma_tag_list
            else:
                morpho_dict[wf.strip()].append((lemma.strip(), tag.strip()))
    return morpho_dict

def get_tag_to_vector():

    tag_to_vector = {} # dictionary mapping POS tags to their numerical vector encoding
    file_location = os.path.dirname(os.path.realpath(__file__))
    f_vectors_mapping = os.path.join(file_location, "tags-vectors-tabs.txt")
    vectors_mapping = open(f_vectors_mapping, "r")
    for line in vectors_mapping.readlines():
        tag, _, vector = line.split("\t")
        tag_to_vector[tag] = vector
    return tag_to_vector

''' For a wordform, get the POS vector representation (which could code ambiguously different features '''
def get_morpho_feats(word, morpho_dict, tag_to_vector):

    if word not in morpho_dict:
        vector_str = "1011011000101111111111111111110000000011001001111111101110011111000000000000"
        vector = vector_str.split()
        vector = np.asarray(map(float, vector))
        return vector
    vector = np.zeros(VECTOR_POSITIONS)
    for _,tag in morpho_dict[word]:
        vector_tmp = np.asarray(map(float, tag_to_vector[tag].split()))
        vector = np.logical_or(vector, vector_tmp)

    return vector