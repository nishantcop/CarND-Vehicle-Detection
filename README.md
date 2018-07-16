## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[classes]: ./output_images/can_non_car.png
[hogNormal]: ./output_images/hog_normal.png
[hogColor]: ./output_images/hog_color.png
[hogPerChannel]: ./output_images/hog_per_channel.png
[slidingSample]: ./output_images/sliding_sample.png
[heatMapCompare]: ./output_images/heat_map_compare.png
[pipelineTest]: ./output_images/pipeline_test.png
[video]: ./output_images/video.gif
---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 5th code cell of the `vehicle_identify.ipynb`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an examples of each of the `vehicle` and `non-vehicle` classes:

![alt text][classes]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

A visualization for car and non_car classes with YCrCb color space

![alt text][hogColor]

A simple visualization for HOG

![alt text][hogNormal]

HOG visualization per channel

![alt text][hogPerChannel]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and features. I finally decided to choose three features: 
1. Color Histogram, choosing color_space to be `YCrCb`
2. Color Channels, I am using three channels to extract HOG features
3. Binned Color Features, `spatial_size =(32, 32)`

Related code is in function `extract_features()` in cell 9. The final set of praameters I used are following: 
 

`color_space = 'YCrCb',
spatial_size = (32, 32),
hist_bins = 32,
orient = 9,
pix_per_cell = 8,
cell_per_block = 2,
hog_channel = 'ALL',
spatial_feat = True,
hist_feat = True,
hog_feat = True`

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using extracted features. Please check cell 12 for training classifier. The training model along with the parameters saved for later usage so that once I am settled with my parameter I don't have to retrain my model. Current test accuracy is 99.1%

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Initially I used a function `find_cars()` to detect the car in cell 19 of `vehicle_identify.ipynb`. This was extracting features using hog sub-sampling and make predictions. The hog sub-sampling helps to reduce calculation time for finding HOG features and thus provided higher throughput rate.

An example with scale `scale` of `1.5` result in following image with a total no of `8 windows` identified. 

![alt text][slidingSample]

Then I used heat map operation to rectify false positive and remove multi detection. For this I tried with different value of hea-map threashold. Following is an example of comparing with two different values i.e. `1` and `0`. Setting the threshold to `1` helped removed false positive. 

![alt text][heatMapCompare]

I've also added three scale window search to to further optimize the false positives or to stablize the car identification that were missing in some of the frames. 

Related code can be found under `cell 23` under section '`Build pipeline`' 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 3 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  I used the following sliding window scales: 

`[(380, 480, 1), (400, 600, 1.5), (500, 700, 2.5)]`

The search is optimize to process frame only once per 10 frames. Also, heat map is optimized by appending 50 pixels for last 3 heat map operations. This helps ease out the wobbly bounding boxes and smooth out the results and further helped to remove unwanted bounded boxes as identified while applying result on test images.  

![alt text][pipelineTest]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/a-uhBAQsSHU)
![alt text][video]
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I still see wobbling rectangles. My aim is to make them super stable. Rectangle border should fit strictly around the vehicle. The one option could be to further optimize the heat map operation. 

Also, in some frames vehicle identification is missing. Possibly need to include larger search area. A possible solution could be to choose for CNN etc. I've choosen linear SVM for the optimization of accouracy Vs. time. 

Still need to impliment `Optional Challange` and `Harder Challange`. Will do that as immediate next step.

