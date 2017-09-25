PLT_AVAILABLE = True
from time import time
import logging
import numpy
import sklearn
import scipy
from tkinter import filedialog
import tkinter as tk
from tkinter import filedialog
try:
	import matplotlib.pyplot as plt
except Exception:
	PLT_AVAILABLE = False
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.externals import joblib
from numpy.random import RandomState
import sys
import numpy as np
import glob
import cv2

def plot_views(images, titles, h, w, n_row=1, n_col=1, ops=None):
	"""Helper function to plot a gallery of portraits"""
	plt.figure(figsize=(3 * 2, 3 * 2))
	plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
	images_shape = images.shape[0]
	if ops != None:
		for i in range(n_row * n_col):        
			if ops > i:
				plt.subplot(n_row, n_col, i + 1)
				plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
				plt.title(titles[i], size=10)

			plt.xticks(())
			plt.yticks(())
	else:
		for i in range(n_row * n_col):        
			if images_shape > i:
				plt.subplot(n_row, n_col, i + 1)
				plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
				plt.title(titles[i], size=10)

			plt.xticks(())
			plt.yticks(())

# plot the result of the prediction on a portion of the test set
def title(y_pred):
	#print(y_pred)
	pred_gender = 'Male' if str(y_pred) == '0' else 'Female'
	return 'predicted: %s\n' % (pred_gender)

def main(img_path, pca_path='pca_model_component_96.jlib', svm_path='svm_model_component_96.jlib'):
	if pca_path == None or len(pca_path) == 0:
		pca_path = 'pca_model_component_96.jlib'
	if svm_path == None or len(svm_path) == 0:
		svm_path='svm_model_component_96.jlib'

	h, w = 192, 168    
	n_components = 96
	flatten_image, input_image, pca, svm = None, None, None, None
	 
	try:
		input_image = cv2.imread(img_path,0)
		input_image = cv2.resize(input_image,(168,192))
		flatten_image = np.array([input_image.flatten()])
		pca = joblib.load(pca_path)
		svm = joblib.load(svm_path)
	except Exception as _e:
		print(_e)
		sys.exit()
	
	eigenfaces = pca.components_.reshape((n_components, h, w))
	#get input image pca
	input_image_pca = pca.transform(flatten_image)
	#predict with input image pca
	Y = svm.predict(input_image_pca)
	Y = 'Male' if Y == '0' else 'Female'
	if PLT_AVAILABLE:
		titles = ['predicted: %s\n' % (Y)]
		eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]        
		plot_views(flatten_image, titles, h, w)
		plot_views(eigenfaces, eigenface_titles, h, w, n_row=4, n_col=4)                
		plt.ion()
		plt.show()
		input('press enter to exit.....')        
		plt.close()

	else:
		print('Predicted',Y)
		cv2.imshow('input image',input_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	

if __name__ == '__main__':
	arg = sys.argv[1::]
	image_path = None
	pca_path = None
	svm_path = None
	root = tk.Tk()
	root.withdraw()
	file_path = filedialog.askopenfilename()

	try:
		with open('config.txt', 'r') as conf:
			c = conf.read().split('\n')
			pca_path = c[0]
			svm_path = c[1]
	except:
		print('config.txt not found, using default param')

	if len(arg) >= 1:
		image_path = arg[0]        
		if len(arg) == 2:
			pca_path = arg[1]
		elif len(arg) == 3:
			pca_path = arg[1]
			svm_path = arg[2]    
	else:
		if file_path:
			image_path = file_path
		else:
			print("closing...", file=sys.stderr)
			sys.exit()           
	
	main(image_path, pca_path, svm_path)
