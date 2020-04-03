from PIL import Image
import csv
from config import config
from datetime import datetime
import matplotlib.pyplot as plt


# Req. 2-2	세팅 값 저장
def save_config():
	with open('./datasets/config.csv', 'a', newline='') as f:
		w = csv.writer(f, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)

		date = datetime.now()
		caption_file_path = config.caption_file_path
		test_size = config.test_size
		random_state = config.random_state
		do_what = config.do_what
		do_sampling = 1 if config.do_sampling else 0
		sampling_size = config.do_sampling

		w.writerow([date, caption_file_path, test_size, random_state, do_what, do_sampling, sampling_size])


# Req. 4-1	이미지와 캡션 시각화
def visualize_img_caption(img_paths, caption):

	# 샘플링 아닌 경우 테스트용 임시 코드
	if not config.do_sampling:
		img_paths = img_paths[:4]
		caption = caption[:4]

	if img_paths.size % 2:
		subplot_size = img_paths.size // 2 + 1
	else:
		subplot_size = img_paths.size // 2

	for i in range(img_paths.size):
		plt.rc('font', size=6)
		plt.subplot(subplot_size, 2, i + 1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		img = Image.open(config.image_file_path + img_paths[i, 0])
		plt.xlabel(caption[i, 0])
		plt.imshow(img)
	plt.show()
