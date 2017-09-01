import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
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
from train_model import get_feature_for_image
from train_model import get_feature_for_image_list
from train_model import get_hog_features
from train_model import bin_spatial
from train_model import color_hist
from train_model import convert_color
from scipy.ndimage.measurements import label


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, nbins, cspace, bins_range):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    # ctrans_tosearch = convert_color(img_tosearch, cspace)
    ctrans_tosearch = img_tosearch
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    box_list = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step


            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], spatial_size)

            # # Extract HOG for this patch
            # hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            # hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            # hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            # hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            # # Get color features
            # spatial_features = bin_spatial(subimg, size=spatial_size)
            # hist_features = color_hist(subimg, nbins=nbins)
            #
            # features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)

            features = get_feature_for_image(subimg, cspace=cspace, size=spatial_size, orient=orient,
                                              pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, nbins=nbins,
                                              bins_range=bins_range).reshape(1, -1)
            test_features = X_scaler.transform(features)
            test_prediction = svc.predict(test_features)


            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                box = ((xbox_left, ytop_draw + ystart),
                       (xbox_left + win_draw, ytop_draw + win_draw + ystart))
                box_list.append(box)
                cv2.rectangle(draw_img, box[0], box[1], (0, 0, 255), 6)

    return draw_img, box_list


def preocess_image(img, ystart, ystop, scale, svc, X_scaler, cspace, spatial_size, orient,
                                              pix_per_cell, cell_per_block, nbins,
                                              bins_range):

    image_with_box, box_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, cspace=cspace, spatial_size=spatial_size, orient=orient,
                                              pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, nbins=nbins,
                                              bins_range=bins_range)
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    heat_image = draw_labeled_bboxes(np.copy(img), labels)
    return image_with_box, heat_image


def batch_process():

    model_path = 'svc_pickle.p'

    dist_pickle = pickle.load(open(model_path, "rb"))
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    nbins = dist_pickle["nbins"]
    cspace = dist_pickle["cspace"]
    bins_range = dist_pickle["bins_range"]

    ystart = 400
    ystop = 656
    scale = 1


    test_images  = glob.glob("test_images/*.jpg")
    for path in test_images:
        print('processing : ', path)
        filename =  os.path.basename(path)
        input_path = path
        box_path = 'output_images/box_' +  filename
        heat_path = 'output_images/heat_' +  filename
        img = mpimg.imread(input_path)
        box_img, heat_img = preocess_image(img, ystart, ystop, scale, svc, X_scaler, cspace, spatial_size, orient,
                                                  pix_per_cell, cell_per_block, nbins,bins_range)
        plt.imsave(box_path, box_img)
        plt.imsave(heat_path, heat_img)



if __name__ == '__main__':
    batch_process()
