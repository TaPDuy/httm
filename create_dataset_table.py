import os
import re
import csv

IMG_PATH = 'prints'

def create_dataset_table():

	with open('data.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(('label', 'path'))

		for img in os.listdir(IMG_PATH):
			label = img[:img.rfind('_')]
			writer.writerow((label, img))

if __name__ == '__main__':
	create_dataset_table()
