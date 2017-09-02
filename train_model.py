import glob
import pickle
import time
import random

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle




def convert_color(image, cspace='RGB'):
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
    else:
        feature_image = image
    return feature_image;


def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel()
    return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 1)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


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


def extract_hog_features(feature_image, orient=9,
                         pix_per_cell=8, cell_per_block=2):
    ch1 = feature_image[:, :, 0]
    ch2 = feature_image[:, :, 1]
    ch3 = feature_image[:, :, 2]
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    return np.hstack((hog1.ravel(), hog2.ravel(), hog3.ravel()))


def get_feature_for_image(image, cspace='RGB', size=(32, 32), orient=9,
                          pix_per_cell=8, cell_per_block=2, nbins=32, bins_range=(0, 256)):
    # image_resize = cv2.resize(image, size)
    feature_image = convert_color(image, cspace=cspace)

    spatial_features = bin_spatial(feature_image, size)
    hist_features = color_hist(feature_image, nbins=nbins, bins_range=bins_range)
    hog_features = extract_hog_features(feature_image, orient=orient,
                                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)

    return np.hstack((spatial_features, hist_features, hog_features))


def get_feature_for_image_list(path_list, cspace='RGB', size=(32, 32), orient=9,
                               pix_per_cell=8, cell_per_block=2, nbins=32, bins_range=(0, 256)):
    features = []
    for path in path_list:
        image = mpimg.imread(path)
        feature = get_feature_for_image(image, cspace=cspace, size=size, orient=orient,
                                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, nbins=nbins,
                                        bins_range=bins_range)
        features.append(feature)
    return features

def show_examples(car_path, non_car_path,  cspace, size, orient,
                                        pix_per_cell, cell_per_block, nbins,
                                        bins_range):
    car_img = mpimg.imread(car_path)
    non_car_img = mpimg.imread(non_car_path)

    mpimg.imsave("output_images/car.png", car_img )
    mpimg.imsave("output_images/non_car.png", non_car_img)

    car_img_color = convert_color(car_img, cspace)
    non_car_img_color = convert_color(non_car_img, cspace)

    mpimg.imsave("output_images/car_color.png", car_img_color )
    mpimg.imsave("output_images/non_car_color.png", non_car_img_color)

    car_img1 = car_img[:, :, 0]
    car_img2 = car_img[:, :, 1]
    car_img3 = car_img[:, :, 2]
    non_car_img1 = non_car_img[:, :, 0]
    non_car_img2 = non_car_img[:, :, 1]
    non_car_img3 = non_car_img[:, :, 2]
    car_feature, car_hog_img1 = get_hog_features(car_img1, orient, pix_per_cell, cell_per_block,
                         vis=True, feature_vec=True)
    car_feature, car_hog_img2 = get_hog_features(car_img2, orient, pix_per_cell, cell_per_block,
                         vis=True, feature_vec=True)
    car_feature, car_hog_img3 = get_hog_features(car_img3, orient, pix_per_cell, cell_per_block,
                         vis=True, feature_vec=True)
    non_car_feature, non_car_hog_img1 = get_hog_features(non_car_img1, orient, pix_per_cell, cell_per_block,
                         vis=True, feature_vec=True)
    non_car_feature, non_car_hog_img2 = get_hog_features(non_car_img2, orient, pix_per_cell, cell_per_block,
                         vis=True, feature_vec=True)
    non_car_feature, non_car_hog_img3 = get_hog_features(non_car_img3, orient, pix_per_cell, cell_per_block,
                         vis=True, feature_vec=True)

    mpimg.imsave("output_images/car_hog1.png", car_hog_img1 )
    mpimg.imsave("output_images/car_hog2.png", car_hog_img2 )
    mpimg.imsave("output_images/car_hog3.png", car_hog_img3 )
    mpimg.imsave("output_images/non_car_hog1.png", non_car_hog_img1)
    mpimg.imsave("output_images/non_car_hog2.png", non_car_hog_img2)
    mpimg.imsave("output_images/non_car_hog3.png", non_car_hog_img3)


def train_model(debug=False):
    t = time.time()

    vehicle_examples = glob.glob("vehicles/*/*.png")
    non_vehicle_examples = glob.glob("non-vehicles/*/*.png")
    #
    # vehicle_examples = vehicle_examples[:1000]
    # non_vehicle_examples = non_vehicle_examples[:1000]
    print(len(vehicle_examples))
    print(len(non_vehicle_examples))


    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    spatial_size = (32, 32)
    nbins = 32
    cspace = 'YCrCb'
    bins_range = (0, 1)

    show_examples(random.choice(vehicle_examples), random.choice(non_vehicle_examples),  cspace=cspace, size=spatial_size, orient=orient,
                                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, nbins=nbins,
                                        bins_range=bins_range)

    if debug:
        return

    car_features = get_feature_for_image_list(vehicle_examples, cspace=cspace, size=spatial_size, orient=orient,
                                              pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, nbins=nbins,
                                              bins_range=bins_range)
    notcar_features = get_feature_for_image_list(non_vehicle_examples, cspace=cspace, size=spatial_size, orient=orient,
                                                 pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, nbins=nbins,
                                                 bins_range=bins_range)
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
        scaled_X, y, test_size=0.1, random_state=rand_state)

    # print('Using:', orient, 'orientations', pix_per_cell,
    #       'pixels per cell and', cell_per_block, 'cells per block')
    print('Training example : ', len(X_train))
    print('Test example : ', len(X_test))

    print('Feature vector length:', len(X_train[0]))
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()

    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Training Accuracy of SVC = ', round(svc.score(X_train, y_train), 4))
    t3 = time.time()
    print(round(t3 - t2, 2), 'Seconds to Evaluate model')

    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    n_predict = 10
    print('Predicts : ', svc.predict(X_test[0:n_predict]))
    print('Labels :   ', y_test[0:n_predict])

    model_path = "svc_pickle.p"

    dist_pickle = {'svc': svc,
                   'scaler': X_scaler,
                   'orient': orient,
                   'pix_per_cell': pix_per_cell,
                   'cell_per_block': cell_per_block,
                   'spatial_size': spatial_size,
                   'nbins': nbins,
                   'cspace': cspace,
                   'bins_range': bins_range
                   }
    with open(model_path, 'wb') as f:
        pickle.dump(dist_pickle, file=f)

    print("Done")


if __name__ == '__main__':
    train_model(debug=True)
