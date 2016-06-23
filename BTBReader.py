import os
import xml.etree.ElementTree as ET


def get_tagged_sentences(input_folder, use_medium_coarse_tagset=False, use_coarsest_tagset=False):

    tag_mapping = "/home/alexander/dev/projects/BAN/resources/morphology/tagset-mapping.xml"
    if (use_medium_coarse_tagset):
        tag_map_dict = {}
        doc = ET.parse(tag_mapping)
        root = doc.getroot()
        pairs = root.findall("pair")
        for pair in pairs:
            specific_tag = pair.find("item").text
            coarse_tag = pair.find("item1").text
            tag_map_dict[specific_tag] = coarse_tag
    if (use_coarsest_tagset):
        tag_map_dict = {}
        doc = ET.parse(tag_mapping)
        root = doc.getroot()
        pairs = root.findall("pair")
        for pair in pairs:
            specific_tag = pair.find("item").text
            coarse_tag = pair.find("item1").text[0]
            tag_map_dict[specific_tag] = coarse_tag



    tagged_sentences = []
    pos_tags = set()
    files = os.listdir(input_folder)
    for f_name in files:
        f = open(os.path.join(input_folder, f_name), "r")
        # remove the .encode/.decode part for NER/POS
        #text = f.read().decode("cp1251")#.encode("utf8")
        text = f.read().decode("utf8")
        sentences = text.split("##")
        for sentence in sentences:
            tagged_sentence = []
            lines = sentence.strip("##").strip().split("\n")
            for line in lines:
                fields = line.split(" ")
                if len(fields) != 2:
                    continue
                word, tag = fields
                tag = tag.strip()
                if (use_medium_coarse_tagset or use_coarsest_tagset) and tag in tag_map_dict:
                    tag = tag_map_dict[tag]
                #else:
                #    tag = tag[0]
                tagged_sentence.append((word, tag))
                pos_tags.add(tag)
            tagged_sentences.append(tagged_sentence)

    return tagged_sentences, pos_tags
