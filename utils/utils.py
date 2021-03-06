import os
import csv
from config import config
from datetime import datetime
import numpy as np


def save_config(base_dir):
	config_path = os.path.join(base_dir, 'config.csv')
	with open(config_path, 'a', newline='') as f:
		w = csv.writer(f, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)

		date = datetime.now()
		caption_file_path = config.caption_file_path
		test_size = config.test_size
		sampling_size = config.do_sampling
		random_state = config.random_state
		val_size = config.val_size
		buffer_size = config.buffer_size
		batch_size = config.batch_size
		epochs = config.epochs

		w.writerow([date, caption_file_path, test_size, sampling_size, random_state,
					val_size, buffer_size, batch_size, epochs])


def map_func(img_name, cap):
	feature_name = os.path.basename(img_name).decode('utf-8').replace('jpg', 'npy')
	img_tensor = np.load((os.path.join(config.base_dir, 'datasets', 'features', feature_name)))
	return img_tensor, cap
