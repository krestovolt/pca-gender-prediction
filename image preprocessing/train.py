from time import time
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.externals import joblib
from numpy.random import RandomState
import numpy as np
import glob
import cv2

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def get_images_path(img_list='image_list.txt'):    
    '''
    input:
    img_list => str 
             => default value: 'image_list.txt'
    description:
    loading image from given image path list, each path followed by ',' then image label (0|1)
    return:
    data => list( (image_path, label) )    
    '''
    data = []
    with open(img_list,'r') as fin:
        for l in fin.read().split('\n'):            
            spl = l.split(',')
            if len(spl) == 2:
                # spl[0] -> image_path
                # spl[1] -> label
                data.append((spl[0],spl[1]))    
    return data

def init_training_data(path_list):
    '''
    input:
    path_list => list( (image_path, label) )
    return:
    X => np.array([flatten_image])
    Y => np.array([label_image])
    img_w => int
    img_h => int
    '''
    X = []
    Y = []
    h, w = cv2.imread(path_list[0][0],0).shape

    for path,label in path_list:
        X.append(cv2.imread(path,0).flatten()) # read in grayscale mode
        Y.append(label)
    print("loaded {} images, {} labels".format(len(X), len(Y)))

    X = np.asarray(X)
    Y = np.asarray(Y)

    return X, Y, w, h

def train_model(n_components = 96, gamma = None, C = None):
    '''
    input:
    n_components => int *must be less than or equal to training data H where H is the number of training data row*
    gamma => <float|int>
    C => <float|int>
    description:
    train new model, if gamma or C are equal None, then the training process use
    grid search for finding best hyperparam, here the classifier model is type of SVM
    after training complete, the SVM model and PCA matrix are automatically saved, if success.
    After training complete, this function will do some evaluation using 20% splited from dataset,
    train the model use 80%.
    output:
    Nothing   
    '''
    data_lpath = get_images_path()
    print('data to read: ',len(data_lpath))
    X, Y, w, h = init_training_data(data_lpath)
    #split 20% of data for testing and the rest for training, choosed randomly after some attempt
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    print("data training/test => {}/{}".format(X_train.shape[0], X_test.shape[0]))

    #PCA part
    n_components = n_components if n_components <= X_train.shape[0] else X_train.shape[0] - abs(X_train.shape[0]-n_components)
    solver = 'randomized'
    print("Extracting the top {} eigenfaces from {} faces".format(n_components, X_train.shape[0]))
    t0 = time()
    pca = PCA(n_components=n_components, svd_solver=solver,whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))

    #eigenface acquired
    eigenfaces = pca.components_.reshape((n_components, h, w))    
    print("Projecting the input data on the eigenfaces")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))

    # build classifier part, this case using svc(Support Vector Classification)
    # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    '''
    The projected image(flatten) against pca
    '''
    clf = None
    t0 = time()

    if gamma == None and C == None:
        # long process, lower or change the variation of hyperparam
        # to test for faster process
        param_grid = [
                    {'C': np.linspace(1, 1e10, num=120),
                    'kernel': ['rbf'],
                    'gamma': np.linspace(1e-5, 0.9, num=120), },
                    {'C': np.linspace(1, 1e10, num=120),
                    'kernel': ['linear'],
                    'gamma': np.linspace(1e-5, 0.9, num=120), },
                    {'C': np.linspace(1, 1e10, num=120),
                    'kernel': ['poly'],
                    'gamma': np.linspace(1e-5, 0.9, num=120), }
                    ]
        try:            
            clf = GridSearchCV(estimator=SVC(verbose=-1, class_weight='balanced', cache_size=500,decision_function_shape='ovr'), param_grid=param_grid, n_jobs=-1)# multi thread
        except Exception as _e:            
            clf = GridSearchCV(estimator=SVC(verbose=False, class_weight='balanced', cache_size=500,decision_function_shape='ovr'), param_grid=param_grid, n_jobs=-1)# multi thread
        clf = clf.fit(X_train_pca, y_train)        
        print("\ndone fit data in %0.3fs" % (time() - t0))
        print("Best estimator found by grid search:")
        print(clf.best_estimator_)        
    else:
        # kernel [rbf, linear, poly, sigmoid]
        # decision_function_shape [ovr, ovo]        
        clf = SVC(gamma=gamma, C=C, decision_function_shape='ovr', class_weight='balanced', verbose=True, kernel='rbf') # not using multi thread
        clf = clf.fit(X_train_pca, y_train)
        print("done fit data in %0.3fs" % (time() - t0))
    
    #post training
    print('='*60)
    print("Eval:")
    print("Predicting gender")
    n_classes = 2
    print("{} classes".format(n_classes))
    t0 = time()
    #get predicted result with input projected test image dataset
    y_pred = clf.predict(X_test_pca)
    print("done in %0.3fs" % (time() - t0))
    print(classification_report(y_test, y_pred))
    '''
        confusion matrix
         e   predicted
         x  
         p 
         e     0   1 
         c 0 | a   b |
         t 1 | c   d |
         e
         d
    '''
    print(confusion_matrix(y_test, y_pred, labels=["0","1"]))
    pca_model_name = "pca_model_component_{}.jlib".format(n_components)
    svm_model_name = "svm_model_component_{}.jlib".format(n_components)

    joblib.dump(pca, pca_model_name)
    joblib.dump(clf, svm_model_name)

    print("done creating new model, saved with name:")
    print("PCA matrix: {}".format(pca_model_name))
    print("SVM model: {}".format(svm_model_name))
    input()

if __name__ == '__main__':	
    conf = input("TRAIN NEW MODEL [ Y | enter to cancel ]>> ")
    if str(conf).lower() == 'y':
        train_model()    