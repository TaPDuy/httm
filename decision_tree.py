import csv
import os
import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import pickle

def load_data(path, img_path):
	data = {
		'label': [],
		'img': [],
	}

	with open(path, 'r') as file:
		reader = csv.reader(file)
		line_cnt = 0
		for label, img in reader:
			if line_cnt > 1:
				im = imread(os.path.join(img_path, img), plugin='pil')
				data['label'].append(label)
				data['img'].append(im)

			line_cnt += 1
	return data

def prepare_training_data(data):
	X = flatten_data(np.array(data['img']))
	y = np.array(data['label'])
	return train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

def flatten_data(data):
	nsamples, nx, ny = data.shape
	return data.reshape((nsamples, nx * ny))

MODEL_PATH = 'models/dtclf.pkl'
def save_model(clf):
	with open(MODEL_PATH, 'wb') as f:
		pickle.dump(clf, f)

USE_LOADED = False
def load_model():
	model = None
	with open(MODEL_PATH, 'rb') as f:
		model = pickle.load(f)
	return model

if __name__ == '__main__':
	data = load_data('data.csv', 'prints')
	X_train, X_test, y_train, y_test = prepare_training_data(data)
	
	dtclf = None
	if not USE_LOADED:
		dtclf = DecisionTreeClassifier()
		dtclf = dtclf.fit(X_train, y_train)
		save_model(dtclf)
	else:
		dtclf = load_model()

	if not dtclf:
		print("Model doesn't exist")
		exit(-1)

	y_pred = dtclf.predict(X_test)
	print(y_test, y_pred)

