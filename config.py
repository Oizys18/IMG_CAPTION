import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument('--caption_file_path', type=str, default=Path(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'captions.csv'), help='캡션경로')
parser.add_argument('--base_dir', type=str, default=Path(os.path.dirname(os.path.abspath(__file__))), help='BASE_DIR')
parser.add_argument('--do_what', type=str, default='train', help='학습(train)을 시킬 것 인지 or 테스트(test)를 할 것인지')
parser.add_argument('--num_words', type=int, default=5000, help='tokenizer 의 크기')
parser.add_argument('--test_size', type=int, default=0.3, help='데이터셋을 테스트데이터로 나눌 비율')
parser.add_argument('--normalize', action='store_false', help='정규화 여부')
parser.add_argument('--do_sampling', type=int, default=0.01, help='샘플링 갯수')
parser.add_argument('--img_aug', action='store_false', help='이미지 데이터 증강')
parser.add_argument('--random_state', type=int, default=918273645, help='shuffle 난수')
parser.add_argument('--val_size', type=int, default=0.2, help='train dataset 을 train 과 val 로 나눌 비율')
parser.add_argument('--embedding_dim', type=int, default=256, help='RNN embedding 차원')
parser.add_argument('--buffer_size', type=int, default=48, help='데이터셋을 섞을 버퍼 크기')
parser.add_argument('--batch_size', type=int, default=16, help='mini batch 크기')
parser.add_argument('--units', type=int, default=512, help='RNN 유닛(unit) 수')
parser.add_argument('--epochs', type=int, default=6, help='EPOCH 크기')
parser.add_argument('--checkpoint', action='store_false', help='체크포인트 로드 여부')
config = parser.parse_args()

print(config.do_what + '_dataset 을 실행시킵니다.')
print(config.img_aug)