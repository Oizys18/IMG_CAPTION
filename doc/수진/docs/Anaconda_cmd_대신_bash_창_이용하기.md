# Anaconda cmd 대신 bash 창 이용하기 

### 1 `.bashrc` 파일을 설정

파일에 직접 쓰기

```.bashrc
# .bashrc
export PATH="$PATH:[Anaconda3 경로]:[Anaconda3 경로]/Scripts"
```

명령어로 쓰기

```cmd
echo 'export PATH="$PATH:[Anaconda3 경로]:[Anaconda3 경로]/Scripts"' >> .bashrc
```

아나콘다의 설치 경로는 `Anaconda` 검색 > 파일 위치 열기 > 속성의 대상칸에서 확인할 수 있다. 



### 2 bashrc 실행

```bash
source .bashrc
```



### 3 conda init

```bash
conda init bash
```

 

✔ 간단한 conda 명령어

```bash
# 가상 환경 만들기 
conda create -n [가상 환경의 이름] python=[버전]
# 가상 환경 삭제 
conda env remove -n [가상 환경의 이름]
# 가상 환경 목록 확인
conda env list
# 가상 환경 활성화
conda activate [가상 환경의 이름]
# 가상 환경 비활성화 
conda deactivate
# 패키지 설치 (한번에 가능)
conda install git matplotlib scikit-learn scikit-learn scipy numpy tensorflow-gpu==2.0.0
# 패키지 삭제 
conda remove -n [가상 환경이름] --all
# 설치된 패키지 확인
conda list 
# 가상 환경 Export
conda env export> [가상 환경의 이름].yaml
# yaml파일을 이용해서 가상 환경 만들기
conda env create -f [가상 환경의 이름].yaml
# 콘다 캐시 삭제
conda clean --all
```



---

참고

[Git과 Anaconda 설치 후 Git Bash 설정](https://azanewta.tistory.com/29)

[ANACONDA CLOUD](https://anaconda.org/anaconda/repo)