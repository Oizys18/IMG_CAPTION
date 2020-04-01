import numpy as np
import tensorflow as tf
import pickle
import json


def save_tokenizer(caption_file_path, caption_num_words):
    data = np.loadtxt(caption_file_path, delimiter='|', dtype=np.str)
    captions = data[1:, 2:]

    captions = np.squeeze(captions, axis=1)
    captions = ['<start>' + cap + ' <end>' for cap in captions]

    top_k = caption_num_words
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k + 1,
                                                      oov_token="<unk>", lower=True,
                                                      split=' ',
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    tokenizer.fit_on_texts(captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    with open('./tokenizer.pickle', 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    tokenizer_json = tokenizer.to_json()
    with open('./tokenizer2.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))


def get_tokenizer():
    with open('../../../datasets/tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

