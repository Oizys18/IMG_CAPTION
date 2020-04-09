import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument('--caption_file_path', type=str, default=Path(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'captions.csv'), help='캡션경로')
parser.add_argument('--image_file_path', type=str, default=Path(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'images'), help='이미지경로')
parser.add_argument('--do_what', type=str, default='train', help='학습(train)을 시킬 것 인지 or 테스트(test)를 할 것인지')
parser.add_argument('--test_size', type=int, default=0.3, help='데이터셋을 테스트데이터로 나눌 비율')
parser.add_argument('--random_state', type=int, default=918273645, help='shuffle 난수')
parser.add_argument('--do_sampling', type=int, default=0.01, help='샘플링 갯수')
parser.add_argument('--normalize', action='store_false', help='정규화 여부')
parser.add_argument('--base_dir', type=str, default=Path(os.path.dirname(os.path.abspath(__file__))), help='BASE_DIR')
parser.add_argument('--img_aug', type=str, default='original', help='이미지 데이터 증강, seq, ba, ba_box')
parser.add_argument('--num_words', type=int, default=5000, help='tokenizer num_words 및 vocab_size')
config = parser.parse_args()

print(config.do_what + '_dataset 을 실행시킵니다.')
print('Image augmentation method: '+config.img_aug)
