import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
import json
import time

dataset_path = '../../../datasets/captions.csv'
data = np.loadtxt(dataset_path, delimiter='|', dtype=np.str)
train_captions = data[1:, 2:]
print('------------train_captions import-------------')
print(train_captions[:5, :])
print()

print(train_captions.shape)
print(train_captions.ndim)
train_captions = np.squeeze(train_captions, axis=1)
stime = time.time()
for i in range(len(train_captions)):
    train_captions[i] = '<start>' + train_captions[i] + ' <end>'
# 0.21904897689819336
# train_captions = ['<start>' + cap + ' <end>' for cap in train_captions]
etime = time.time()
# 0.17506980895996094
print(etime-stime)
print()
print('------------add <start> and <end> token-------------')
print(train_captions[:5])
print(train_captions.shape)
print(train_captions.ndim)
print()


"""
Preprocess and tokenize the captions
First, you'll tokenize the captions (for example, by splitting on spaces). 
    This gives us a vocabulary of all of the unique words in the data (for example, "surfing", "football", and so on).
Next, you'll limit the vocabulary size to the top 5,000 words (to save memory). 
    You'll replace all other words with the token "UNK" (unknown).
You then create word-to-index and index-to-word mappings.
Finally, you pad all sequences to be the same length as the longest one.
"""


# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)


# Choose the top 5000 words from the vocabulary
top_k = 5000
# num_words :  빈도수가 높은 상위 몇 개의 단어만 사용하겠다고 지정
# + 1 : padding 때문에
# oov : 케라스 토크나이저는 기본적으로 단어 집합에 없는 단어인 OOV에 대해서는
# 단어를 정수로 바꾸는 과정에서 아예 단어를 제거한다는 특징이 있습니다.
# 단어 집합에 없는 단어들은 OOV로 간주하여 보존하고 싶다면 Tokenizer의 인자 oov_token을 사용합니다.
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k + 1,
                                                  oov_token="<unk>", lower=True,
                                                  split=' ',
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')


tokenizer.fit_on_texts(train_captions)  # 빈도수를 기준으로 단어 집합을 생성

print(tokenizer.word_index)  # 각 단어에 인덱스가 어떻게 부여되었는지
print('%s개의 고유한 토큰을 찾았습니다.' % len(tokenizer.word_index))
# {'<unk>': 1, 'a': 2, 'in': 3, 'on': 4, 'the': 5, 'of': 6, 'man': 7, 'with': 8,}
print(tokenizer.word_counts)  # 각 단어가 카운트를 수행하였을 때 몇 개였는지
# OrderedDict([('two', 20), ('young', 10), ('guys', 4), ('with', 23), ('shaggy', 1), ('hair', 2),])
# print(tokenizer.word_index['<unk>'])
# print(tokenizer.word_index['<start>'])
# print(tokenizer.index_word[3])
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'
# tokenizer.word_index['<start>'] = 1
# tokenizer.index_word[1] = '<start>'
# tokenizer.word_index['<end>'] = 2
# tokenizer.index_word[2] = '<end>'
# print(tokenizer.word_index['<start>'])
# print(tokenizer.index_word[0], tokenizer.index_word[3])

# Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)  # 각 단어를 이미 정해진 인덱스로 변환
print(train_seqs[:3])

# Pad each vector to the max_length of the captions
# If you do not provide a max_length value, pad_sequences calculates it automatically
# padding: String, 'pre' or 'post': pad either before or after each sequence.
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

# Calculates the max_length, which is used to store the attention weights
max_length = calc_max_length(train_seqs)

# Create training and validation sets using an 80-20 split
cap_train, cap_val = train_test_split(cap_vector,
                                      test_size=0.2,
                                      random_state=0)
print(cap_vector[:3])
# print(cap_train[:3])
# print(cap_train.shape)
# print(len(cap_train), len(cap_val))
# print()
# print(cap_train[:5, :])


# pickle 파일로 저장하기
with open('./tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)


# with open('tokenizer.pickle', 'rb') as f:
#     h = pickle.load(f)


# JSON file 로 저장하기
tokenizer_json = tokenizer.to_json()
with open('./tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))
