from pytorch_lightning import LightningDataModule, LightningModule, Trainer

def tag_sentence(model:LightningModule,
                 device:str,
                 sentence:str,
                 tokenizer,
                 text_field,
                 tag_field):
    """
    Tag a sentence with tags using the trained model
    :param model:
    :param device:
    :param sentence:
    :param tokenizer:
    :param text_field:
    :param tag_field:
    :return:
    """
    model.eval()
    if isinstance(sentence, str):
        tokens = tokenizer.tokenize(sentence)
    else:
        tokens = sentence
    numericalized_tokens = tokenizer.convert_tokens_to_ids(tokens)
    numericalized_tokens = [text_field.init_token] + numericalized_tokens
    unk_idx = text_field.unk_token
    unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
    token_tensor = torch.LongTensor(numericalized_tokens)
    token_tensor = token_tensor.unsqueeze(-1).to(device)
    predictions = model(token_tensor)
    top_predictions = predictions.argmax(-1)
    predicted_tags = [tag_field.vocab.itos[t.item()] for t in top_predictions]
    predicted_tags = predicted_tags[1:]
    assert len(tokens) == len(predicted_tags)
    return tokens, predicted_tags, unks