import glob
import os
import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

import vehicle

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

x_start_stop = (300, 1280)
y_start_stop = (400, 656)
xy_window = (64, 64)
xy_overlap = (0.5, 0.5)
scale = 1.5
y_start_stop_window=(
                    ((400, 530), 64),
                    ((400, 600), 96),
                    ((400, 656), 128))

def batch_process():
    test_images  = glob.glob("test_images/*.jpg")
    for path in test_images:
        print('processing : ', path)
        filename =  os.path.basename(path)
        input_path = path
        box_path = 'output_images/box_' +  filename
        heat_path = 'output_images/heat_' +  filename
        img = mpimg.imread(input_path)

        vehicleDetector = vehicle.VehicleDetector(x_start_stop, y_start_stop, xy_window, xy_overlap, svc, X_scaler,
                                                  cspace, spatial_size, orient,
                                                  pix_per_cell, cell_per_block, nbins, bins_range, scale,
                                                  y_start_stop_window)
        box_img, heat_img = vehicleDetector.detect(img, include_box_image=True)
        plt.imsave(box_path, box_img)
        plt.imsave(heat_path, heat_img)


def video_process():
    vehicleDetector = vehicle.VehicleDetector(x_start_stop, y_start_stop, xy_window,xy_overlap, svc, X_scaler, cspace, spatial_size, orient,
                                                  pix_per_cell, cell_per_block, nbins,bins_range, scale, y_start_stop_window)
    output_video = 'project_video_output.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    # output_video = 'test_video_output.mp4'
    # clip1 = VideoFileClip("test_video.mp4")


    output_clip = clip1.fl_image(vehicleDetector.detect)
    output_clip.write_videofile(output_video, audio=False)

if __name__ == '__main__':
    video_process()
    # batch_process()
