import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# 데이터 로드 (현동 코드)
def get_data_file():
    data = np.load('../../datasets/test_datasets.npy')
    img_paths = data[:50, :1]
    captions = data[:50, 2:]
    train_images = np.squeeze(img_paths, axis=1)
    train_images = ['../../datasets/images/' + img for img in train_images]
    train_captions = np.squeeze(captions, axis=1)
    train_captions = ['<start>' + cap + ' <end>' for cap in train_captions]
    train_images = list(set(train_images))
    print(train_images[:3])
    print(train_captions[:3])
    return train_images, train_captions


def img_normalization_1(img):
    image = tf.constant(img, dtype=tf.float32)
    mean, var = tf.nn.moments(image, axes=[0, 1, 2])
    centered = (image - mean) / var**0.5
    # print((image - mean) / var**0.5)
    plt.imshow(centered)
    plt.show()


def img_normalization_2(img):
    image = tf.constant(img, dtype=tf.float32)
    centered = tf.image.per_image_standardization(image)
    # print(centered)
    plt.imshow(centered)
    plt.show()


def image_load(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (255, 255))
    return img, image_path

img_name_vector, train_captions = get_data_file()
print(len(img_name_vector))


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

train_images, train_captions = get_data_file()
for timg in train_images:
    img, image_path = image_load(timg)
    img_normalization_2(img)


"""
image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# Get unique images
encode_train = sorted(set(img_name_vector))

# Feel free to change batch_size according to your system configuration
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

for img, path in tqdm(image_dataset):
  batch_features = image_features_extract_model(img)
  batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))

  for bf, p in zip(batch_features, path):
    path_of_feature = p.numpy().decode("utf-8")
    np.save(path_of_feature, bf.numpy())

# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

# Choose the top 5000 words from the vocabulary
top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
train_seqs = tokenizer.texts_to_sequences(train_captions)

tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'
"""
