import numpy as np
import tensorflow as tf
import pickle


def get_captions(dataset_path):
    data = np.loadtxt(dataset_path, delimiter='|', dtype=np.str)
    data = data[:, 2:]
    return np.squeeze(data, axis=1)


def save_tokenizer(captions, caption_num_words=5000):
    test_captions = ['<start>' + cap + ' <end>' for cap in captions]

    top_k = caption_num_words
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k + 1,
                                                      oov_token='<unk>', lower=True,
                                                      split=' ',
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    tokenizer.fit_on_texts(test_captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    print('%s개의 토큰을 가진 Tokenizer 를 저장합니다.' % len(tokenizer.word_index))
    with open('./datasets/tokenizer_sample.pkl', 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)


def change_word_to_vector(captions):
    with open('./datasets/tokenizer_sample.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    train_seqs = tokenizer.texts_to_sequences(captions)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    print('각 caption 을 토큰화 한 결과를 보여줍니다.')
    print(cap_vector[:2])


caption_dataset = get_captions('./datasets/test_datasets.csv')
save_tokenizer(caption_dataset)
print()
change_word_to_vector(caption_dataset)
