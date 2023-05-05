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

tag2color = {
    "ADJ": "background-color: blue; color: white;",
    "ADP": "background-color: green; color: white;",
    "ADV": "background-color: red; color: white;",
    "AUX": "background-color: purple; color: white;",
    "CCONJ": "background-color: brown; color: white;",
    "DET": "background-color: darkblue; color: white;",
    "INTJ": "background-color: magenta; color: white;",
    "NOUN": "background-color: darkgreen; color: white;",
    "NUM": "background-color: navy; color: white;",
    "PART": "background-color: orange; color: white;",
    "PRON": "background-color: darkorange; color: white;",
    "PROPN": "background-color: darkred; color: white;",
    "PUNCT": "background-color: black; color: white;",
    "SCONJ": "background-color: darkmagenta; color: white;",
    "SYM": "background-color: darkcyan; color: white;",
    "VERB": "background-color: darkviolet; color: white;",
    "X": "background-color: darkgray; color: white;"
}

def generate_color_legend():
    html_legend = "<div><h4>Color Legend:</h4>"
    for tag, style in tag2color.items():
        html_legend += f'<span style="margin-right: 15px; {style}">{tag}</span>'
    html_legend += "</div>"
    return html_legend
id2tag = {v: k for k, v in tag2id.items()}

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