import tensorflow as tf
import os
import numpy as np
from data.preprocess import load_image
from models.encoder import CNN_Encoder
from models.decoder import RNN_Decoder
import pickle
from train import embedding_dim, units, vocab_size, img_name_train, img_name_val, max_length, cap_val
import matplotlib.pyplot as plt
from PIL import Image
from config import config
from pathlib import Path


encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)
num_steps = len(img_name_train)


BASE_DIR = os.path.join(config.base_dir, 'datasets')
tokenizer_path = os.path.join(BASE_DIR, 'tokenizer.pkl')
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

attention_features_shape = 64


def evaluate(image):
    image_model = tf.keras.applications.InceptionV3(
        include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(
        temp_input)
    img_tensor_val = tf.reshape(
        img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(
            dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


rid = np.random.randint(0, len(img_name_val))
image = f"{Path(config.base_dir,'datasets','images',img_name_val[rid])}"
real_caption = ' '.join([tokenizer.index_word[i]
                         for i in cap_val[rid] if i not in [0]])

result, attention_plot = evaluate(image)
print(img_name_val[rid])
print('Real Caption:', real_caption)
print('Prediction Caption:', ' '.join(result))
plot_attention(image, result, attention_plot)

img = Image.open(image)
w, h = img.size
plt.text(0, h+50, 'Real Caption: {}\nPrediction Caption: {}'.format(real_caption, ' '.join(result)), fontsize=12)
plt.imshow(img)
plt.show()
