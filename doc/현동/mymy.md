```
conda install -c conda-forge notebook
jupyter notebook

conda activate AI
```

https://towardsdatascience.com/practical-coding-in-tensorflow-2-0-fafd2d3863f6



-  
-  20200402

https://www.tensorflow.org/tutorials/text/word_embeddings

https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map

https://github.com/GunhoChoi/PyTorch-FastCampus/blob/master/05_RNN/0_Basic/Simple_Char_RNNcell.ipynb

https://wikidocs.net/book/2155

https://www.tensorflow.org/tutorials/text/image_captioning



- Allocator (GPU_0_bfc) ran out of memory trying to allocate  https://eehoeskrap.tistory.com/360 

- Tennsorflow v2 Limit GPU Memory usage https://github.com/tensorflow/tensorflow/issues/25138



- numpy 차원 확장 및 축소 https://076923.github.io/posts/Python-numpy-9/

- Image captioning with visual attention https://www.tensorflow.org/tutorials/text/image_captioning

- tokenizer https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer





- 정수 인코딩 https://wikidocs.net/31766

- 텍스트 시퀀스 https://subinium.github.io/Keras-6-1/

- LSTM https://datascienceschool.net/view-notebook/770f59f6f7cc40c8b6dc98dddd06c6c5/

- RNN https://excelsior-cjh.tistory.com/183?category=940400

- 토큰화 https://wikidocs.net/21698



- 모도를 위한 텐서훌로우 https://www.youtube.com/watch?v=L69jcg3A2_k&list=PLQ28Nx3M4Jrguyuwg4xe9d9t2XE639e5C&index=43



- tanc http://taewan.kim/post/tanh_diff/
- https://www.google.com/search?q=tanh&oq=tanh&aqs=chrome..69i57j0l7.3320j0j7&sourceid=chrome&ie=UTF-8





- master
  - hotfix
- release
  - bugfix
- develop
  - feature







**Sub2 스프린트#2 프로젝트 마무리** `~16:30`

1. Jira > Issues 완료된 이슈(이슈 > 미해결 이슈 > Epic 포함)만 완료로 처리 하기
2. 스프린트 완료 처리하기(미해결 이슈는 다음 백로그로 자동 이동됨-계획이 왜 잘못되었는지도 고민)
3. 코드 정리 (Polish code)불필요한 내용들 정리 및 .gitignore 에 등록빌드된 폴더: node_modules, dist, build, target, checkpoints로그/임시파일, .settings, .idea, **pycache** 등등...Lint 적용: flake8, pep8사용되지 않는 import, 변수, 함수, class 정리
4. 코드 리펙토링폴더구조, 파일명, 클래스명이 명확한지 확인반복/제어문이 깊어지면(indent) 함수로 분리하기주석이 작성되었다면 변수나 함수명을 명확히 해서 주석이 없이도 이해할 수 있도록 하기
5. 프로젝트 사용법 작성(README.md)Overview-프로젝트 설명, Prerequisites, Installation, Usage(Pretrained model, Training, Inference)방법들 등등..License 설정작성 참고: https://github.com/jcjohnson/densecap
6. 최종 산출물Gitlab(Sub1 Req관련 프로젝트 소스, READEME.md)

**스프린트 검토회의(고객에게 제품을 데모시연,리뷰)** `16:30 ~ 17:00`

- 팀별 랜덤으로 진행예정
- develop 에 머지에서 시연

**제품 출시** `17:00 ~`

1. 기본 브랜치 master 로 변경 후 develop -> master 브랜치로 머지하기
2. "master" 브랜치에 version tagging(ex: v1.0.0), 내용에 릴리즈 노트 작성하기 노트작성 참고: https://github.com/spring-projects/spring-boot/releases
3. Gitlab 프로젝트 정보 업데이트: 설정 > 일반 > General projectProject description : 프로젝트 한줄 설명Tags: 프로젝트 태깅 ex) python, ai, tensorflow, deep-learning, rnn, cnn, image-captioning

**팀내 스프린트 회고회의** `~18:00`

- 팀장 주도하에 줌을 통해 회의를 개설하여 각자 이번 프로젝트시 1.잘하거나 좋았던 점, 2. 힘들었거나 아쉬웠던 점, 3. 개선할 점 말하고 메러모스트 팀방에 남기기
- 서기1명이 Gitlab 프로젝트에 해당 내용들 취합해서 Gitlab > Wiki 에 마크다운으로 위 3가지 항목으로 남기기





