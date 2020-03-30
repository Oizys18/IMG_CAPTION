# README

- SSAFY 2기 서울 4반 5팀 
- Sub02



## Dev_Notice

#### Branch

- master 

- release

- develop  
  - feature branches
  - front / back / ML / DB 
  - ex) develp >> back >> `feature/auth/login`

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

  ![](C:\Users\jayhy\OneDrive\바탕 화면\commit1.PNG)

  ![](C:\Users\jayhy\OneDrive\바탕 화면\commit2.PNG)

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

