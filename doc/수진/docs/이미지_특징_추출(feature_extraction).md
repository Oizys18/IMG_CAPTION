## feature extraction 

이미지 특징 값 추출을 위해 ImageNet 으로 사전 학습 된 Network layer를 사용하도록 한다. 

아래 코드는 tensorflow 공식 튜토리얼 - [Image captioning with visual attention](https://www.tensorflow.org/tutorials/text/image_captioning)을 참고하여 작성

#### 캡션과 이미지 경로 불러오기

- train 과 test가 각각 다른 `npy` 파일로 저장 된 상태

- 추후 `config` 파일에서  train, test mode 에 따라 경로를 바꿔주는 설정 필요

  (테스트를 위해 50개로만 돌려보는중 !)

```python 
from ai_sub2 import config

def get_data_file():
    data = np.load(os.path.join(config.base_dir, 'datasets', 'test_datasets.npy'))
    img_paths = data[:50, :1]
    captions = data[:50, 2:]
    ##### np.shape(img_paths) = (50, 1)
    train_images = np.squeeze(img_paths, axis=1)
    ##### np.shape(train_images) = (50, )
    train_images = [os.path.join(config.base_dir, 'datasets', 'images', f'{img}.jpg') for img in train_images]
    train_captions = np.squeeze(captions, axis=1)
    train_captions = ['<start>' + cap + ' <end>' for cap in train_captions]
    return train_images, train_captions

```



#### 이미지 파일 불러오기

- inceptioin_V3는 기본 이미지 사이즈가 299 * 299 
- [preprocess_input](https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/preprocess_input) 로 전처리 과정을 수행 
  - 이미지 픽셀을 -1 ~ 1 범위 의 값으로 정규화 
  - Inception V3의 학습에 주로 이용된 형식으로 match
- 마지막 convolutional layer에서 이미지 특징값을 추출 할 수 있다.

```python
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path
```



#### Inception V3 Weights 값 초기화하고 ImageNet으로 사전 학습 된 Weight 불러오기

- [tf.keras.applications.InceptionV3](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3?version=nightly)

- output layer 는 `(8, 8, 2048)`

- 해당 예제에서 attention을 사용했기 때문에 마지막 conv layer를 사용

  *Visual Attention*

  Visual Attention은 이미지의 특정 부분에 집중하는 것. 

- 속도상(bottleneck)의 문제로 모델링은 최초 1번만 수행함

- 따라서 최초의 값을 dictionary로 저장해둔다. (image_name -> feature_vector)

```python
image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
```



#### 추출한 이미지 특징 Caching 하기

- tqdm을 이용해서 output을 RAM 에 Caching 한다.
- 추후 batch 값 config에 저장
  - batch: 한번에 처리할 image의 갯수
- `num_parallel_calls=tf.data.experimental.AUTOTUNE` : 병렬데이터셋을 불러오고, 파일을 여는 데 기다리는 시간을 단축 : [tf.dataAPI로 성능 향상하기](https://www.tensorflow.org/guide/data_performance?hl=ko)

```python
encode_train = sorted(set(img_name_vector))


image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

for img, path in tqdm(image_dataset):
    batch_features = image_features_extract_model(img)

    batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))
    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        print(os.path.basename(path_of_feature))
        np.save(os.path.join(config.base_dir, 'datasets','features',os.path.basename(path_of_feature).replace('jpg', 'npy')), bf.numpy())# 캡션, augmentation으로 중복 이미지 발생.
encode_train = sorted(set(img_name_vector))

# batch 값을 조정
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

for img, path in image_dataset:
  batch_features = image_features_extract_model(img)
  batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))

  for bf, p in zip(batch_features, path):
    path_of_feature = p.numpy().decode("utf-8")
    np.save(path_of_feature, bf.numpy())
```



✔ 사용법

​	config에서 모드 설정, 갯수 설정 후 feature_extraction.py 실행!

#### Python 경로 관리  [참고](https://itmining.tistory.com/122)

```python
import os
# 현재 작업 폴더 얻기 
os.getcwd()
# 디렉토리 변경
os.chdir('[path]')
# 특정 경로에 대해 절대 경로 얻기
os.path.abspath('[path]')
# 경로 중 디렉토리명만 얻기
os.path.dir(path)
# 경로 중 파일명만 얻기
os.path.basename(path)
# 경로를 병합하여 새 경로 생성
os.path.join('[path]','[path1]','[path2]')
```



---

참고

[tensorflow slim API를 이용한 Network 구현](http://blog.naver.com/PostView.nhn?blogId=laonple&logNo=221399862902&parentCategoryNo=&categoryNo=&viewDate=&isShowPopularPosts=false&from=postView)

 [Inception V3 Retraining 하기](http://solarisailab.com/archives/1422)

