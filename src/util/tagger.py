tag2id = {
    "ADJ": 0,
    "ADP": 1,
    "ADV": 2,
    "AUX": 3,
    "CCONJ": 4,
    "DET": 5,
    "INTJ": 6,
    "NOUN": 7,
    "NUM": 8,
    "PART": 9,
    "PRON": 10,
    "PROPN": 11,
    "PUNCT": 12,
    "SCONJ": 13,
    "SYM": 14,
    "VERB": 15,
    "X": 16
}

def align_tags(tags, offsets_mapping):
    aligned_tags = []
    tag_idx = 0
    for offset in offsets_mapping:
        if offset[0] == 0 and offset[1] != 0:  # If the token is not a special token or a subword
            aligned_tags.append(tag2id[tags[tag_idx]])
            tag_idx += 1
        else:
            aligned_tags.append(-100)  # Use -100 to mask the loss for this token
    return aligned_tags