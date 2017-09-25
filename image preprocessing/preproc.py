import cv2
import glob
import os
from makelist import makelist

def preproc(debug = False, terminal = False):
	currentFile = __file__
	p = '/'.join(os.path.realpath(currentFile).replace('\\','/').split('/')[:-1])
	ps = [glob.glob(p+'/data/0/*.jpg'),glob.glob(p+'/data/1/*.jpg')]	
	print('working on',p)	

	for i in range(2):
		images = ps[i]
		len(images)		
		b = 256
		k = 3
		l = 5
		m = 7
		for image in images:
			img = cv2.imread(image,0)
			img = cv2.equalizeHist(img)
			img = cv2.resize(img,(168,192))

			img = img[::] + 1.8

			gk = cv2.GaussianBlur(img,(k,k),0)
			image_p = ''.join(image.split(".")[0:len(image)-1])+"_Gaussian_"+str(k)
			image_t = "."+image.split(".")[-1]
			if debug:
				print(image_p+image_t)
			path = image_p+image_t			
			cv2.imwrite(path.replace('\\','/').replace('data','preprocessed'),gk)

		del images

	if not terminal:
		makelist()

	return True

if __name__ == "__main__":
	if preproc():
		makelist()