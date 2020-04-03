## Dev_Notice

#### updated - 2020.04.02

- Mod config.py
  - config.py 에 각 변수마다 default 값을 할당해두었습니다.
  - train.py 파일 실행시 terminal 에서 `python train.py` 만 입력하면 됩니다. 
  - 주어진 captions.csv 파일을 불러와서 train / val dataset 으로 분리하는 과정(sub01)은 실행되지 않습니다. (주석처리)
  - 미리 만들어 둔 train_dataset.npy 파일만 읽어오고 실행시킵니다.
  - config.csv 파일은 datasets 폴더에 그대로 두었습니다. gitignore 에 등록해두었습니다.

- Add train_dataset.npy && val_dataset.npy 
  - captions.csv 파일을 7:3 비율로 **섞지않고**(이미지 1개+캡션 5개 고정) 나눠서 만들어두었습니다. 
  - 자료 구조를 확인할 수 있게 csv 파일도 함께 만들어두었습니다.

- Add tokenizer.pkl
  - train, test_dataset 과 마찬가지로 sub02 - req 2 에 따른 정적파일입니다. 
- 요약
  - train_dataset.npy, test_dataset.npy, tokenizer.pkl 과 같이 datasets 에 있는 파일은 고정입니니다!
  - 따로 이야기가 나오기 전까지 수정, 변경사항이 있으면 꼭 공유해주세요~

+

- 가상환경 설정

  - 본인의 가상환경 삭제
    conda env remove -n [가상 환경의 이름]
  - 리스트 확인
    conda env list
  - 찬우의 가상환경으로 설정
    conda env create -f [가상 환경의 이름].yaml

  

---



#### Branch

- master 

- release

- develop  
  - feature branches
  - front / back / ML / DB 
  - ex) develp >> back >> feature: `back/auth/login`

- developer
  - personal branches - 개인적으로 필요시 파일 공유, 저장 및 관리 
  - ex) `soulG/conf`



#### Commit message

- https://blog.ull.im/engineering/2019/03/10/logs-on-git.html 참고

- Convention 

  - 기본
    - modify  : 코드 수정
    - add  : 새 파일 작성
    - delete : 코드 및 파일 삭제
  - 전면 수정
    - refactor  
  - 버전 관리
    - update

- shift enter 누르고 내용을 작성하면 아래처럼 `...` 로 요약 된 커밋 메시지를 남길 수 있습니다~

  ![](C:/Users/jayhy/OneDrive/바탕 화면/Home/SSAFY_Project/03_특화프로젝트/sub02/s02p22a405/doc/images/commit1.PNG)

  ![](C:/Users/jayhy/OneDrive/바탕 화면/Home/SSAFY_Project/03_특화프로젝트/sub02/s02p22a405/doc/images/commit2.PNG)

  ```
  $ commit -m 'message file.exe feature
  [shift enter]
  > detail message (여기는 한글 or 영어)'
  ```

  ```bash
  'modify users.py login  
  로그인시 쿠키 발급 추가
  '
  ```

