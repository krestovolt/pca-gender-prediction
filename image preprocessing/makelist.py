import glob
import os

def makelist():
	currentFile = __file__
	p = '/'.join(os.path.realpath(currentFile).replace('\\','/').split('/')[:-1])
	print('working on',p)
	MALE = 0
	FEMALE = 1

	female_images = glob.glob(p+"/preprocessed/1/*.jpg")
	male_images = glob.glob(p+"/preprocessed/0/*.jpg")

	print("training data:")
	print("FEMALE :{} images found".format(len(female_images)))
	print("MALE :{} images found".format(len(male_images)))

	with open(p+"/image_list.txt","w") as file_out:
		for file in female_images:
			file_out.write("%s,%d\n"%(file.replace('\\','/'), FEMALE))
		for file in male_images:
			file_out.write("%s,%d\n"%(file.replace('\\','/'), MALE))

if __name__ == "__main__":
	makelist()