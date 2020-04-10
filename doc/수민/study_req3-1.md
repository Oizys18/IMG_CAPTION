# Req 3. Dataset 생성

## Reference

- [Image captioning with visual attention](https://www.tensorflow.org/tutorials/text/image_captioning)

## Req 3-1. `tf.data.Dataset` 생성

- **이미지 파일 경로** 와 **캡션** 을 입력으로 받아 **이미지 데이터** 및 **토큰화된 캡션 쌍** 을 리턴하는 함수를 구현한다.
  - 단, 리턴 값은 텐서플로우 데이터 형식 tf.data.Dataset 을 따르도록 한다.

---

- Req2 결과값 `cap_vector` 의 data type 은 `numpy.ndarray` 이다.

  - 예

    ```python
    array([[   3,    2,  351, ...,    0,    0,    0],
           [   3,    2,   31, ...,    0,    0,    0],
           [   3,    2,  318, ...,    0,    0,    0],
           ...,
           [   3,    2,  201, ...,    0,    0,    0],
           [   3, 1462,  264, ...,    0,    0,    0],
           [   3,    2,  175, ...,    0,    0,    0]], dtype=int32)
    ```

- train 용 / test 용 데이터를 분할하는 함수에서 리턴값의 data type은 각각 `<class 'list'>` `<class 'numpy.ndarray'>` 이다.

  ```python
  img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                      cap_vector,
                                                                      test_size=0.2,
                                                                      random_state=0)
  ```

  - img_name_train 예

    ```python
    ['./dataset/images/__________.jpg',
     './dataset/images/__________.jpg',
     './dataset/images/__________.jpg',
     './dataset/images/__________.jpg',
     './dataset/images/__________.jpg',]
    ```

  - `cap_train` 예

    ```python
    array([[  3,  29,  53, ...,   0,   0,   0],
           [  3,   1, 212, ...,   0,   0,   0],
           [  3,   2,  31, ...,   0,   0,   0],
           ...,
           [  3,   2,  44, ...,   0,   0,   0],
           [  3,   2, 530, ...,   0,   0,   0],
           [  3,   2, 659, ...,   0,   0,   0]], dtype=int32)
    ```

### Create a tf.data.Dataset for training

#### 이미지 데이터 관련 세팅

```python
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = top_k + 1 # 5000 + 1
num_steps = len(img_name_train)

# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64
```

#### `tf.data.Dataset`

```python
dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
print(dataset)
```

```bash
 <TensorSliceDataset shapes: ((), (49,)), types: (tf.string, tf.int32)>
```

```python
# Load the numpy files
def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor, cap
```

- 명세의 **1. img_name** 과 **2. cap**을 입력으로 받아 **1. img_tensor(이미지 데이터)** 와 **2. cap (토큰화된 캡션)** 을 반환하는 함수에 해당한다.

```python
# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
```

```bash
<ParallelMapDataset shapes: (<unknown>, <unknown>), types: (tf.float32, tf.int32)>
```

```pypthon
# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
```

```bash
<PrefetchDataset shapes: (<unknown>, <unknown>), types: (tf.float32, tf.int32)>
```
