import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    # Use cv2.resize().ravel() to create the feature vector
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_color_features(image, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    # Iterate through the list of images
    # Read in each one by one
    # apply color conversion if other than 'RGB'

    features = []
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)
    # Apply bin_spatial() to get spatial color features
    spatial_features = bin_spatial(feature_image, color_space=cspace, size=spatial_size)
    # Apply color_hist() also with a color space option now
    hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
    # Append the new feature vector to the features list
    features.extend(spatial_features)
    features.extend(hist_features)
    return features


def get_feature_for_image_list(path_list):
    features = []
    for path in path_list:
        image = mpimg.imread(path)
        features.append(get_feature_for_image(image))
    return features;


def get_feature_for_image(image, size=(32, 32)):
    image_resize = cv2.resize(image, size)

    features = []
    features.extend(extract_hog_features(image_resize))
    features.extend(extract_color_features(image_resize))
    return features


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_hog_features(image, cspace='RGB', orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel='ALL'):
    # Create a list to append feature vectors to
    # Iterate through the list of images
    # Read in each one by one
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)

    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel],
                                orient, pix_per_cell, cell_per_block,
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)
    else:
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    # Append the new feature vector to the features list
    return hog_features

def train_model():
    t = time.time()

    vehicle_examples = glob.glob("vehicles/*/*.png")
    non_vehicle_examples = glob.glob("non-vehicles/*/*.png")
    print(len(vehicle_examples))
    print(len(non_vehicle_examples))
    #
    # vehicle_examples = vehicle_examples[:100]
    # non_vehicle_examples = non_vehicle_examples[:100]

    car_features = get_feature_for_image_list(vehicle_examples)
    notcar_features = get_feature_for_image_list(non_vehicle_examples)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to extract HOG features...')
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    rand_state = np.random.randint(0, 100)

    scaled_X, y = shuffle(scaled_X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    # print('Using:', orient, 'orientations', pix_per_cell,
    #       'pixels per cell and', cell_per_block, 'cells per block')
    print('Training example : ', len(X_train))
    print('Test example : ', len(X_test))

    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    # parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    # svr = SVC()
    # clf = grid_search.GridSearchCV(svr, parameters)
    # clf.fit(iris.data, iris.target)

    svc = SVC(kernel='linear')
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()

    # C_range = 10. ** np.arange(-3, 8)
    # gamma_range = 10. ** np.arange(-5, 4)
    #
    # param_grid = dict(gamma=gamma_range, C=C_range)
    #
    # grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(y=y_train, k=5))
    #
    # grid.fit(X_train, y_train)
    #
    # print("The best classifier is: ", grid.best_estimator_)
    # svc = grid.best_estimator_
    # plot the scores of the grid
    # grid_scores_ contains parameter settings and scores
    # score_dict = grid.grid_scores_
    # print(" Best score ", score_dict)
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Training Accuracy of SVC = ', round(svc.score(X_train, y_train), 4))
    t3 = time.time()
    print(round(t3 - t2, 2), 'Seconds to Evaluate model')

    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t4 = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])

    # joblib.dump(svc, 'model.pkl')
    model_path = "svc_pickle.p"

    orient = 9
    pix_per_cell = 8
    cell_per_block = 2


    spatial_size = (32, 32)
    hist_bins =32


    dist_pickle = {'svc': svc,
                   'scaler': X_scaler,
                   'orient':orient,
                    'pix_per_cell': pix_per_cell,
                   'cell_per_block':cell_per_block,
                   'spatial_size': spatial_size,
                   'hist_bins': hist_bins
                   }
    with open(model_path, 'wb') as f:
        pickle.dump(dist_pickle, file=f)

    print("Done")

def test_model():
    vehicle_examples = glob.glob("vehicles/*/*.png")
    non_vehicle_examples = glob.glob("non-vehicles/*/*.png")
    print(len(vehicle_examples))
    print(len(non_vehicle_examples))
    #
    # vehicle_examples = vehicle_examples[:100]
    # non_vehicle_examples = non_vehicle_examples[:100]

    car_features = get_feature_for_image_list(vehicle_examples)
    notcar_features = get_feature_for_image_list(non_vehicle_examples)
    t2 = time.time()
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    rand_state = np.random.randint(0, 100)

    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    svc = joblib.load('model.pkl')
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))



if __name__ == '__main__':
    train_model()
