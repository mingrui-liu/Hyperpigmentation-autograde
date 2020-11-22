import tensorflow as tf
import glob
import os
import pandas as pd
import randaugment
from sklearn.model_selection import train_test_split

import constants


AUTOTUNE = tf.data.experimental.AUTOTUNE

image_size = constants.IMAGE_SIZE
BATCH_SIZE = constants.BATCH_SIZE
data_path = "./Unilever/"

def create_generators():

	def normalize(x):
		x /= 127.5
		x -= 1.
		return x

	def load_and_preprocess_image(path):
		image = tf.io.read_file(path)
		image = tf.image.decode_jpeg(image, channels=3)
		image = tf.image.resize(image, [image_size, image_size])
		image = tf.clip_by_value(image, 0.0, 255.0)
		image = tf.cast(image, dtype=tf.uint8)

		return image

	def load_file_from_disk(data_path):
		all_image_paths = sorted(glob.glob(data_path + "image_test/*.jpg"))

		# read the excel file
		MHPgrades = pd.read_excel(data_path + "MHPgrades.xlsx")
		grade = MHPgrades["Mottled hyperpigmentation"]

		#group and generate the lable
		all_image_labels = list(grade)
		all_label = []
		for grade in all_image_labels:
			if grade < 2.0:
				label = 1.5
			elif grade > 4.0:
				label = 4.5
			else:
				label = grade
			all_label.append(int(label * 2))

		label_to_index = dict((name, index) for index, name in enumerate(sorted(set(all_label))))
		all_labels = [label_to_index[grade] for grade in all_label]
		              
		# train test split 
		train_paths, test_paths, train_labels, test_labels = train_test_split(all_image_paths, all_labels,test_size = 0.05)

		return train_paths, test_paths, train_labels, test_labels



	train_paths, test_paths, train_labels, test_labels = load_file_from_disk(data_path)

	train_paths = tf.data.Dataset.from_tensor_slices(train_paths)
	image_ds = train_paths.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

	label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_labels, tf.float32))
	image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))


	# here's our final training dataset
	train_ds = image_label_ds.cache()
	train_ds = train_ds.shuffle(buffer_size = 1024)
	train_ds = train_ds.repeat()
	train_ds = train_ds.map(lambda image, label: (tf.image.random_flip_left_right(image),label),num_parallel_calls=AUTOTUNE)
	train_ds = train_ds.map(lambda image,label:(randaugment.distort_image_with_randaugment(image,num_layers = 2,magnitude = 2),label),num_parallel_calls=AUTOTUNE)
	train_ds = train_ds.map(lambda image,label:(normalize(tf.cast(image,tf.float32)),label),num_parallel_calls=AUTOTUNE)

	train_ds = train_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)


	path_ds = tf.data.Dataset.from_tensor_slices(test_paths)
	image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

	label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(test_labels, tf.float32))
	test_ds = tf.data.Dataset.zip((image_ds, label_ds))
	test_ds= test_ds.map(lambda image,label:(normalize(tf.cast(image,tf.float32)),label),num_parallel_calls=AUTOTUNE)
	
	test_ds = test_ds.cache().batch(BATCH_SIZE)

	return train_ds,test_ds



