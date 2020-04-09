from config import config
from data import preprocess
from utils import utils
from data.feature_extraction import feature_extraction
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from models.encoder import CNN_Encoder
from models.decoder import RNN_Decoder
import time

tf.autograph.experimental.do_not_convert()
tf.compat.v1.reset_default_graph()


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(img_tensor, target):
    loss = 0
    hidden = decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += loss_function(target[:, i], predictions)
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


BASE_DIR = os.path.join(config.base_dir, 'datasets')
tokenizer_path = os.path.join(BASE_DIR, 'tokenizer.pkl')
checkpoint_path = os.path.join(config.base_dir, 'checkpoints')

# config 저장
utils.save_config(BASE_DIR)

# dataset 을 train 과 test datasets 으로 구분 및 저장
preprocess.dataset_split_save(BASE_DIR, config.test_size)

# tokenizer 호출
tokenizer = preprocess.get_tokenizer(tokenizer_path, config.num_words)

# 사용자의 설정에 따라서 train or test dataset 불러오기
img_name_vector, train_captions = preprocess.get_data_file(BASE_DIR,
                                                           config.do_what,
                                                           config.do_sampling)

# image_augmentation 및 pre_trained model 실행
img_name_vector, train_captions = feature_extraction(BASE_DIR,
                                                     img_name_vector,
                                                     train_captions,
                                                     config.img_aug)

# 각 caption 을 token 벡터로 변경
cap_vector, max_length = preprocess.change_text_to_token(tokenizer_path, train_captions)

# 불러온 dataset 을 train 과 val dataset 으로 구분
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=config.val_size,
                                                                    random_state=config.random_state)  

# hyper parameter 설정
EPOCHS = config.epochs
BUFFER_SIZE = config.buffer_size
BATCH_SIZE = config.batch_size
embedding_dim = config.embedding_dim
units = config.units
vocab_size = config.num_words + 1
num_steps = len(img_name_train)

# 학습을 위한 데이터셋 설정 (tf.data dataset)
tf.compat.v1.enable_eager_execution()
dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# 이미지 특징 벡터 불러오기
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          utils.map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch, buffer 설정
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# # CNN RNN 모델
# encoder = CNN_Encoder(embedding_dim)
# decoder = RNN_Decoder(embedding_dim, units, vocab_size)
#
# # optimizer & loss object
# optimizer = tf.keras.optimizers.Adam()
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
#                                                             reduction='none')
#
# # 체크포인트 매니저 생성
# ckpt = tf.train.Checkpoint(encoder=encoder,
#                            decoder=decoder,
#                            optimizer=optimizer)
# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)
#
#
# # train
# start_epoch = 0
# if ckpt_manager.latest_checkpoint:
#     start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
#     print("마지막으로 저장된 epoch {} 부터 시작합니다.".format(start_epoch + 1))
#     ckpt.restore(ckpt_manager.latest_checkpoint)
#
# loss_plot = []
# for epoch in range(start_epoch, EPOCHS):
#     start = time.time()
#     total_loss = 0
#
#     for (batch, (img_tensor, target)) in enumerate(dataset):
#         batch_loss, t_loss = train_step(img_tensor, target)
#         total_loss += t_loss
#
#         if batch % 100 == 0:
#             print('Epoch {} Batch {} Loss {:.4f}'.format(
#               epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
#
#     loss_plot.append(total_loss / num_steps)
#
#     if epoch % 5 == 0:
#         ckpt_manager.save()
#         ckpt_path = ckpt_manager.save()
#         print('Epoch {} 저장 : {}'.format(epoch+1, ckpt_path))
#
#     print('Epoch {} Loss {:.6f}'.format(epoch + 1,
#                                         total_loss/num_steps))
#     print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
