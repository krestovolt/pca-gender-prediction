####################################################################################################################################################################################

-- execute RUN.bat at image preprocessing for producing preprocessed grayscale image and generate new pca matrix and svm classifier.

-- execute RUN.bat at gender prediction for running trained svm classifier with generated pca matrix from above process.

dependency:
numpy -> matrix/vector/array manipulation, flatten image and reshaping, creating array of number
opencv -> image manipulation and reader
scipy(sklearn) -> svm + other utility for loggin and evaluating the model, grid search for searching best svm hyperparameter
matplotlib(pyplot) -> visualization
tkinter(filedialog) -> open file dialog

python ver. used for development: python 3.5 64-bit

images source:
1. Kuang-Chih Lee, Jeffrey Ho, and David Kriegman in "Acquiring Linear Subspaces for Face Recognition under Variable Lighting, PAMI, May, 2005
   =>http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html

2. from Gil Levi and Tal Hassner research on Age and Gender Classification using Convolutional Neural Networks, 
   IEEE Workshop on Analysis and Modeling of Faces and Gestures (AMFG) 
   => http://www.openu.ac.il/home/hassner/Adience/data.html



####################################################################################################################################################################################

							Kautsar Assyifa B P 			Yusuf Fathony Putrasandi 