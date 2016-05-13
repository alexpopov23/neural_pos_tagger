import os


def get_morpho_dict(path_to_dict):

    morpho_dict = {}
    files = os.listdir(path_to_dict)
    for f_path in files:
        f = open(os.path.join(path_to_dict, f_path), "r")
        lines = f.readlines()
        for line in lines:
            wf, tag, lemma = line.split("\t")
            if wf not in morpho_dict:
                lemmas = []
                lemmas.append(lemma.strip())
                morpho_dict[wf.strip()] = lemmas
            else:
                morpho_dict[wf.strip()].append(lemma.strip())
    return morpho_dict
