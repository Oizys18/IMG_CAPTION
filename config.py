import argparse

# Req. 2-1	Config.py 파일 생성
parser = argparse.ArgumentParser()

parser.add_argument('--caption_file_path', type=str, default='./datasets/captions.csv', help='캡션경로')
parser.add_argument('--image_file_path', type=str, default='./datasets/images/', help='이미지경로')
parser.add_argument('--do_what', type=str, required=True, help='학습(train)을 시킬 것 인지 or 테스트(val)를 할 것인지')
parser.add_argument('--test_size', type=int, default=0.4, help='테스트로 나눌 비율')
parser.add_argument('--random_state', type=int, default=1234, help='shuffle 난수')
parser.add_argument('--do_sampling', type=int, help='샘플링 갯수')
config = parser.parse_args()

print(config.do_what + '_dataset 을 실행시킵니다.')
