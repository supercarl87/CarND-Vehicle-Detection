## CarND-Vehicle-Detection


---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png

[car]: ./output_images/car.png
[non_car]: ./output_images/non_car.png
[car_color]: ./output_images/car_color.png
[non_car_color]: ./output_images/non_car_color.png
[car_hog1]: ./output_images/car_hog1.png
[car_hog2]: ./output_images/car_hog2.png
[car_hog3]: ./output_images/car_hog3.png

[non_car_hog1]: ./output_images/non_car_hog1.png
[non_car_hog2]: ./output_images/non_car_hog2.png
[non_car_hog3]: ./output_images/non_car_hog3.png
[box_test1]: ./output_images/box_test1.jpg
[box_test2]: ./output_images/box_test2.jpg
[box_test3]: ./output_images/box_test3.jpg
[box_test4]: ./output_images/box_test4.jpg
[box_test5]: ./output_images/box_test5.jpg
[heat_test1]: ./output_images/heat_test1.jpg
[heat_test2]: ./output_images/heat_test2.jpg
[heat_test3]: ./output_images/heat_test3.jpg
[heat_test4]: ./output_images/heat_test4.jpg
[heat_test5]: ./output_images/heat_test5.jpg
[car]: ./output_images/car.png



[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this part is in `train_model.py` for `get_hog_features` and `extract_hog_features`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

* car image

![alt text][car]

* non car image

![alt text][non_car]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

* YCrCb color car
![alt text][car_color]
* YCrCb color non car
![alt text][non_car_color]

* Car hog1  
![alt text][car_hog1]
* Car hog2  
![alt text][car_hog2]
* Car hog3  
![alt text][car_hog3]

* Non Car hog1  
![alt text][non_car_hog1]
* Non Car hog2  
![alt text][non_car_hog2]
* Non Car hog3  
![alt text][non_car_hog3]



#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and class default shows good test accuracy.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using scikit. I use both histograms color feature and also bin feature. I split the input data with 90% training data and 10% test data. I shuffle the data before training to make sure the order of the data will not affect the classifier performance.  Ultimately, this can achieve 98.63% test accuracy.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I have two sliding window approaches implemented in the code. One approach will take in x y position and generate all windows based on window sides and generate features based on sub image. The window will be overlap 50%.

The second approach is use scaled window. Since calculate hog feature is slow and we do not need to calculate each time for every sub image. This approach will generate hog image once based on the scale and take the sub array for features.
I use different windows side in different region. `64,64` for middle region and `96,96` and `128,128` for larger region.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  
For performance optimization, we use faster sliding window approach mentioned in above section.


Here are some example images:

![alt text][box_test1]
![alt text][box_test2]
![alt text][box_test3]
![alt text][box_test4]
![alt text][box_test5]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

To make the window more stable in video, I recorded last 8 bounding box found in the previous video frames and apply this to new the existing bounding box, so the current frame can untilize the information in previous detected cars.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Examples with box combined with heat map :

![alt text][heat_test1]
![alt text][heat_test2]
![alt text][heat_test3]
![alt text][heat_test4]
![alt text][heat_test5]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

[improved]The sliding window size is sensitive to detect larger or small cars，to optimize nearby cars, I used larger windows side.  The current pipeline will more likely to fail on small car in remote range. We could use different window size based on the y value, since nearby car will appear to be larger.

The current pipeline use 8 frame of historical data, it will be stable after certain frame, but it does not perform good when new car just entered.

The pipeline was very slow since we generate hog feature for every sub image, and it is faster with hog image once, but it is still slow. One way to explore could be using gray scale image which will capture most of feature and reduce the hog calculate.

### References
[CarND Q&A][https://www.youtube.com/watch?v=P2zwrTM8ueA&list=PLAwxTw4SYaPkz3HerxrHlu1Seq8ZA7-5P&index=5]
