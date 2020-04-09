### Optimizer

학습속도를 빠르고 안정적이게 하는 것 : [참고](https://gomguard.tistory.com/187)

SGD, Momentum, NAG, Adagrad, Adadelta ,Rmsprop 등이 있다.

![1](.\img\optimizer발전.JPG)



```pyton
optimizer = tf.keras.optimizers.Adam()
```

### Loss Function 

```python
# loss 모델 만들기 (SparseCategoricalCrossentropy)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

# loss function 
def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)
```



### train_step

크게 아래 세 가지 step으로 나눌 수 있다.

1.  오차 구하기 
2. gradient 구하기 
3. weight 업데이트 

`train_step`

```python
@tf.function
def train_step(img_tensor, target):
  loss = 0

  # initializing the hidden state for each batch
  # because the captions are not related from image to image
  hidden = decoder.reset_state(batch_size=target.shape[0])

  dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

  with tf.GradientTape() as tape:
      features = encoder(img_tensor)

      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, hidden, _ = decoder(dec_input, features, hidden)

          loss += loss_function(target[:, i], predictions)

          # using teacher forcing
          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)

  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss
```



- [tf.function](https://www.tensorflow.org/guide/function?hl=ko) 데코레이터 : tensor 그래프 내에서 컴파일 되었을 때, GPU나 TPU를 사용해서 작동, 세이브드 모델로 내보내는 것이 가능함

- 데코레이터가 붙은 함수로부터 호출 된 함수들은 그래프 모드에서 동작함.

`train`

```python
EPOCHS = 20

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(
              epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)

    if epoch % 5 == 0:
      ckpt_manager.save()

    print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                         total_loss/num_steps))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
```

- epoch 

  학습용 사진 전체를 한 번 사용하면 한 세대가 지나감. (1 epoch)

  1 epoch에 이용되는 훈련 데이터는 여러개의 batch로 분할됨.