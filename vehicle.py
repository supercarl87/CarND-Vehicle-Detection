import cv2
import numpy as np
from scipy.ndimage.measurements import label
from collections import deque

from train_model import bin_spatial
from train_model import color_hist
from train_model import convert_color
from train_model import get_feature_for_image
from train_model import get_hog_features


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
    bbox_list = []
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        bbox_list.append(bbox)
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img, bbox_list


def slide_window(x_start_stop, y_start_stop, xy_window, xy_overlap):
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))

    nx_windows = np.int(xspan / nx_pix_per_step) - 1
    ny_windows = np.int(yspan / ny_pix_per_step) - 1
    window_list = []

    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            window_list.append(((startx, starty), (endx, endy)))
    return window_list


def find_cars_fast(img, xstart_stop, ystart_stop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                   hist_bins, window):
    img = img.astype(np.float32) / 255
    ystart = ystart_stop[0]
    xstart = xstart_stop[0]
    img_tosearch = img[ystart_stop[0]:ystart_stop[1], xstart_stop[0]:xstart_stop[1], :]
    ctrans_tosearch = convert_color(img_tosearch, cspace='YCrCb')
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
    # window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
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
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                box = ((xbox_left + xstart, ytop_draw + ystart),
                       (xbox_left + win_draw + xstart, ytop_draw + win_draw + ystart))
                box_list.append(box)

    return box_list


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, x_start_stop, y_start_stop, xy_window, xy_overlap, svc, X_scaler, orient, pix_per_cell,
              cell_per_block, spatial_size, nbins, cspace, bins_range):
    img = img.astype(np.float32) / 255
    windows = slide_window(x_start_stop, y_start_stop, xy_window, xy_overlap)

    box_list = []

    for window in windows:
        subimg = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        features = get_feature_for_image(subimg, cspace=cspace, size=spatial_size, orient=orient,
                                         pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, nbins=nbins,
                                         bins_range=bins_range).reshape(1, -1)
        test_features = X_scaler.transform(features)
        test_prediction = svc.predict(test_features)

        if test_prediction == 1:
            box = window
            box_list.append(box)

    return box_list


def box_draw_box_image(img, box_list):
    draw_img = np.copy(img)
    for box in box_list:
        cv2.rectangle(draw_img, box[0], box[1], (0, 0, 255), 6)
    return draw_img


def preocess_image(img, scale, x_start_stop, y_start_stop, xy_window, xy_overlap, svc, X_scaler, cspace, spatial_size,
                   orient,
                   pix_per_cell, cell_per_block, nbins,
                   bins_range, history, y_start_stop_window, include_box_image=False):
    box_list = []

    # box_list_partial = find_cars_fast(img, y_start_stop[0], y_start_stop[1], scale, svc, X_scaler,
    #                                                   orient,
    #                                                   pix_per_cell, cell_per_block, spatial_size, nbins, 64)
    # box_list.extend(box_list_partial)
    for (y_start_stop, window) in y_start_stop_window:
        if window ==64 :
            box_list_partial = find_cars_fast(img, x_start_stop,  y_start_stop, scale, svc, X_scaler, orient,
                                              pix_per_cell, cell_per_block, spatial_size, nbins, window)
        else:
            box_list_partial = find_cars(img, x_start_stop, y_start_stop, (window, window), xy_overlap, svc, X_scaler, orient, pix_per_cell,
              cell_per_block, spatial_size, nbins, cspace, bins_range)
        box_list.extend(box_list_partial)
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)

    for item in history:
        box_list.extend(item)
    # Add heat to each box in box list
    heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 2)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)


    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    heat_image, bbox_list = draw_labeled_bboxes(np.copy(img), labels)

    history.append(bbox_list)
    if len(history) >= history.maxlen:
        history.pop()


    if include_box_image:
        image_with_box = box_draw_box_image(np.copy(img), box_list)
        return image_with_box, heat_image
    else:
        return heat_image


class VehicleDetector:
    history = deque(maxlen=8)
    processed =0
    def __init__(self, x_start_stop, y_start_stop, xy_window, xy_overlap, svc, X_scaler, cspace, spatial_size, orient,
                 pix_per_cell, cell_per_block, nbins, bins_range, scale, y_start_stop_window):
        self.x_start_stop = x_start_stop
        self.y_start_stop = y_start_stop
        self.xy_window = xy_window
        self.xy_overlap = xy_overlap
        self.svc = svc
        self.X_scaler = X_scaler
        self.cspace = cspace
        self.spatial_size = spatial_size
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.nbins = nbins
        self.bins_range = bins_range
        self.scale = scale
        self.y_start_stop_window = y_start_stop_window

    def detect(self, image, include_box_image=False):
        # self.processed += 1
        # if self.processed < 200 or self.processed > 300:
        #     return image
        if include_box_image:
            history = deque(maxlen=8)
        return preocess_image(image, self.scale, self.x_start_stop, self.y_start_stop, self.xy_window, self.xy_overlap,
                              self.svc, self.X_scaler, self.cspace,
                              self.spatial_size, self.orient,
                              self.pix_per_cell, self.cell_per_block, self.nbins, self.bins_range, self.history, self.y_start_stop_window,
                              include_box_image=include_box_image)
