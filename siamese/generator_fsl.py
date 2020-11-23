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
			if grade < 3.0:
				label = 2.5
			elif grade > 5.0:
				label = 5.5
			else:
				label = grade
			all_label.append(int(label * 2))

		label_to_index = dict((name, index) for index, name in enumerate(sorted(set(all_label))))
		all_labels = [label_to_index[grade] for grade in all_label]
		              
		# train test split 
		train_paths, test_paths, train_labels, test_labels = train_test_split(all_image_paths, all_labels, test_size=0.1, random_state=1)
		train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=0.1, random_state=1)

		return train_paths,val_paths, test_paths, train_labels, val_labels, test_labels


	def load_file_from_disk_with_additional_image(data_pat):
		all_image_paths_1 = sorted(glob.glob(data_path + "image_test/*.jpg"))
		all_image_paths_2 = sorted(glob.glob(data_path + "Additional_Images/*.jpg"))

		# read the excel file
		MHPgrades = pd.read_excel(data_path +"MHPgrades.xlsx",header = 0)
		MHPgrades2 = pd.read_excel(data_path +"SKN-APP-0452-MHPgrades.xlsx",header = 1)

		MHPgrades2 = MHPgrades2.dropna(axis = 0)
		MHPgrades2 = MHPgrades2.melt(id_vars=["Subject No.", "Side of Face"], 
		        var_name="VISIT", 
		        value_name="Mottled hyperpigmentation").sort_values(by = ['Subject No.','Side of Face'])

		grade1 = MHPgrades["Mottled hyperpigmentation"]

		#Manipulate excel file
		MHPgrades2.loc[MHPgrades2['VISIT'] == 'Baseline' , 'VISIT'] = '00'
		MHPgrades2.loc[MHPgrades2['VISIT'] == 'Week 4' , 'VISIT'] = '04'
		MHPgrades2.loc[MHPgrades2['VISIT'] == 'Week 8' , 'VISIT'] = '08'
		MHPgrades2.loc[MHPgrades2['VISIT'] == 'Week 12' , 'VISIT'] = '12'
		MHPgrades2.loc[MHPgrades2['VISIT'] == 'Week 16' , 'VISIT'] = '16'
		MHPgrades2.loc[MHPgrades2['VISIT'] == 'Week 17' , 'VISIT'] = '17'

		MHPgrades2.loc[MHPgrades2['Side of Face'] == 'Left' , 'Side of Face'] = 'L'
		MHPgrades2.loc[MHPgrades2['Side of Face'] == 'Right' , 'Side of Face'] = 'R'

		#match grade and images and make valid pair
		valid_path = []
		valid_label = []
		for index, row in MHPgrades2.iterrows():
		    id = str(row['Subject No.'] )
		    id = id.zfill(4)

		    side = row['Side of Face']
		   
		    visit = str(row['VISIT'])

		    grade = row['Mottled hyperpigmentation']
		  


		    output = [path for path in all_image_paths_2 if path.startswith('/content/drive/My Drive/Unilever/Additional_Images/P' + id + '_FACE_' + side + '_S00_VCR_S2_V00D00W' + visit)] 
		    if len(output)>0:
		      valid_path.append(output[0])
		      valid_label.append(grade)

		all_image_paths = all_image_paths_1 + valid_path
		all_image_labels = list(grade1) +  valid_label

		##Create label and zip dataset
		all_label = []
		for grade in all_image_labels:
		  if grade < 3.0:
		    label = 2.5
		  elif grade > 5.0:
		    label = 5.5
		  else:
		    label = grade
		  all_label.append(label)

		label_to_index = dict((name, index) for index, name in enumerate(sorted(set(all_label))))
		all_labels = [label_to_index[grade] for grade in all_label]

		##train val test split
		train_paths, test_paths, train_labels, test_labels = train_test_split(all_image_paths, all_labels, test_size=0.1, random_state=1)
		train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=0.1, random_state=1)

		return train_paths,val_paths, test_paths, train_labels, val_labels, test_labels





	def create_ds(paths,labels):
		train_paths_1 = tf.data.Dataset.from_tensor_slices(paths)
		train_paths_2 = tf.data.Dataset.from_tensor_slices(paths)
		image_ds_1 = train_paths_1.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
		image_ds_2 = train_paths_2.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

		label_ds_1 = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.float32))
		label_ds_2 = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.float32))



		image_label_ds_1 = tf.data.Dataset.zip((image_ds_1, label_ds_1))
		image_label_ds_1 = image_label_ds_1.shuffle(64)

		image_label_ds_2 = tf.data.Dataset.zip((image_ds_2, label_ds_2))
		image_label_ds_2 = image_label_ds_2.shuffle(64)
		

		image_label_ds = tf.data.Dataset.zip((image_label_ds_1, image_label_ds_2))

		return image_label_ds

	def detect(label1,label2):

		label = tf.cond(tf.equal(label1,label2), lambda: tf.constant(1.0), lambda: tf.constant(0.0))

		return label




	train_paths,val_paths, test_paths, train_labels, val_labels, test_labels = load_file_from_disk_with_additional_image(data_path)
  # here's our final training dataset
	train_ds = create_ds(train_paths,train_labels)
	val_ds = create_ds(val_paths,val_labels)
	test_ds = create_ds(test_paths,test_labels)


  #train
	train_ds = train_ds.cache()
	train_ds = train_ds.shuffle(buffer_size = 1024)
	train_ds = train_ds.repeat()
	train_ds = train_ds.map(lambda image_label1,image_label2: (tf.image.random_flip_left_right(image_label1[0]),tf.image.random_flip_left_right(image_label2[0]),image_label1[1],image_label2[1]),num_parallel_calls=AUTOTUNE)
	#train_ds = train_ds.map(lambda image1,image2,label1,label2:(randaugment.distort_image_with_randaugment(image1,num_layers = 1,magnitude = 2),
	#															randaugment.distort_image_with_randaugment(image2,num_layers = 1,magnitude = 2),
	#															label1,
	#															label2),num_parallel_calls=AUTOTUNE)
	train_ds = train_ds.map(lambda image1,image2,label1,label2 :(normalize(tf.cast(image1,tf.float32)),normalize(tf.cast(image2,tf.float32)),label1,label2),num_parallel_calls=AUTOTUNE)
	train_ds = train_ds.map(lambda image1,image2,label1,label2 :((image1,image2),detect(label1,label2)),num_parallel_calls=AUTOTUNE)
	train_ds = train_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

  #val
	val_ds = val_ds.cache()
	val_ds = val_ds.shuffle(buffer_size = 64)
	val_ds = val_ds.repeat()
	val_ds = val_ds.map(lambda image_label1,image_label2: (image_label1[0],image_label2[0],image_label1[1],image_label2[1]),num_parallel_calls=AUTOTUNE)
	val_ds = val_ds.map(lambda image1,image2,label1,label2 :(normalize(tf.cast(image1,tf.float32)),normalize(tf.cast(image2,tf.float32)),label1,label2),num_parallel_calls=AUTOTUNE)
	val_ds = val_ds.map(lambda image1,image2,label1,label2 :((image1,image2),detect(label1,label2)),num_parallel_calls=AUTOTUNE)
	val_ds = val_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

  #test
	test_ds = test_ds.cache()
	test_ds = test_ds.shuffle(buffer_size = 64)
	test_ds = test_ds.repeat()
	test_ds = test_ds.map(lambda image_label1,image_label2: (image_label1[0],image_label2[0],image_label1[1],image_label2[1]),num_parallel_calls=AUTOTUNE)
	test_ds = test_ds.map(lambda image1,image2,label1,label2 :(normalize(tf.cast(image1,tf.float32)),normalize(tf.cast(image2,tf.float32)),label1,label2),num_parallel_calls=AUTOTUNE)
	test_ds = test_ds.map(lambda image1,image2,label1,label2 :((image1,image2),detect(label1,label2)),num_parallel_calls=AUTOTUNE)
	test_ds = test_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)








	return train_ds,val_ds, test_ds,test_labels

