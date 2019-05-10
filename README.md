# CarND-Capstone-Project
## Programming a Real Self Driving Car

---

[//]: # (Image References)

[image1]: ./support/structure.png "Program Structure"
[image2]: ./support/SimulatorStartUp.png "Simulator Welcome"
[image3]: ./support/SimlatorStarted.png "Simulator Running"
[image4]: ./support/FullyRunning.png "Fully Running"


- [Team: Autowheels](#sec-1)
- [Overview](#sec-2)
- [System Architecture](#sec-3)
- [Installation steps](#sec-4)
- [Usage](#sec-5)
- [Traffic Light Detection and Classification End-to-End Approach using Tensorflow](#sec-6)
  - [Development Overview](#sec-6-1)
  - [Performance Evaluation](#sec-6-2)
  - [PID Tuning Parameters](#sec-6-3)
- [Results](#sec-7)
- [License](#sec-8)

---

## Meet Team AutoWheels<a id="sec-1"></a>
Team AutoWheels has five members. Below are their names, email addresses and slack handles.


Team Member Name | Email Address | Slack Handle 
------------ | ------------- | -------------
Diogo Silva (Team Lead) | akins.daos+selfdriving@gmail.com | @diogoaos
Volker van Aken | volker.van.aken@gmail.com | @Volker
Andreea Patachi | patachiandreea@yahoo.com | @Andreea	
Stephen Nutman | stephen.nutman@ntc-europe.co.uk | @Steve
Alexander Meade | alexander.n.meade@gmail.com | @ameade

## Overview<a id="sec-2"></a>

This is the final project in the Udacity Self Driving Car NanoDegree course. The task of this Capstone project was to create ROS nodes to implement core functionality of an autonomous vehicle system, including traffic light detection, vehicle control and waypoint path following. The development uses a simulator to support in evaluating the code performance. Once ready to run there was an opportunity to run the code on a real car - the Udacity AD vehicle Carla.

## System Architecture Diagram<a id="sec-3"></a>

The following system diagram shows the architecture of the code that was implemented. The architecture is split into 3 main areas:

- Perception (Traffic Light Detection)
- Planning (Waypoint Following)
- Control (Vehicle longitudinal and lateral control)

From this diagram the ROS topics can be seen communicating between the ROS nodes. Information is also passed on these topics to the Car simulator.


![alt text][image1]


## Installation steps<a id="sec-4"></a>

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed:[One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

## Usage<a id="sec-5"></a>

1. Make a project directory `mkdir project_udacity && cd project_udacity`
2. Clone this repository into the project_udacity directory. `https://github.com/nutmas/CarND-Capstone.git`
3. Install python dependencies. `cd CarND-Capstone-Project\` and `pip install -r requirements.txt` will install dependencies.
4. Build code. `cd ros\` and `catkin_make` and `source devel/setup.sh`
5. Create a directory for simulator `cd` and `mkdir Sim` and `cd Sim`
6. Download Simulator from here: [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases)
7. Run the simulator `cd linux_sys_int` and `./sys_int.x86_64` (for linux 64bit system)

![alt text][image2]


8. Launch the code. `cd CarND-Capstone-Project\ros\` and `roslaunch launch\styx.launch`

![alt text][image3]

9. Clicking the `Camera` checkbox will ready the car for autonomous mode. A green planned path appears.

![alt text][image4]

10. Now the vehicle is ready to drive autonomously around the track. Click the `Manual` checkbox and the vehicle will start to drive.


## Traffic Light Detection and Classification End-to-End Approach using Tensorflow<a id="sec-6"></a>

#### Development Overview<a id="sec-6-1"></a>

An end-to-end approach in the traffic light detection context equates to passing the classifier an image; it then identifies the location in the scene
and also categorises the traffic light state as RED, YELLOW or GREEN.

To achieve this I decided to develop a network model by retraining an existing model from the Tensorflow [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

The models selected were deemed suitable for the traffic light task, based on performance and output:  
**faster_rcnn_inception_v2_coco** Speed: 60ms, Quality: 28mAP, Output: Boxes  
**faster_rcnn_resnet101_coco** Speed: 106ms, Quality: 32mAP, Output: Boxes

The following process was utilised to retrain the models to enable them to classify traffic lights in the simulator.

- Drive around simulator track and log images received from camera on rostopic /image_color. To get a range of traffic light conditions 3 Laps of track data was gathered.
- A dataset was compiled using [labelimg](https://github.com/tzutalin/labelImg). Bounding boxes were drawn around the front facing traffic lights, and labelled as RED, YELLOW, GREEN or UNKOWN. Images with no traffic lights were not labelled.
- The Object Detection libraries in Tensorflow v1.12 were required to enable re-training of the models. The dataset was converted to a tensorflow 'record' to proceed with training.
- The basic configuration for each model in the training setup is:
    - Inception v1: Epoch: 2000 Input Dimensions: min:600 max:800
    - Inception v2: Epoch: 20000 Input Dimensions: min:600 max:800
    - Resnet: Epoch: 80000 Input Dimensions: min:600 max:800
- Training the models was performed using the scripts available in the Tensorflow Object library.
- I created python-notebook pipeline to test each model against a set of images which the model had not seen during training. The notebook painted bounding boxes on each image, providing the classification and confidence. 500 images passed through produced the results for Inception v2 are shown in this [Video](https://www.youtube.com/watch?v=1QT6ahoyVDY&t=124s)
- After successful static image evaluation all models were frozen; For compatibility with Udacity environment freezing was performed using Tensorflow v1.4.
- The frozen models were integrated into the [`tl_classifier.py`](https://github.com/nutmas/CapstoneProject-AutoWheels/blob/TensorBranch/ros/src/tl_detector/light_classification/tl_classifier.py) node of the pipeline.  
    - From ROS camera image is received by 'tl_detector.py' and passed into a shared lockable variable.
    - The function `get_classification()` is ran in a parallel thread to process the image and utilise the classifier. This avoids the classifier impacting on the ROS processing its other tasks.
    - The classifier processes the image and returns the detection and classification results.
    - The array of classification scores for each traffic light detection are evaluated and highest confidence classification is taken as the result to pass back to [`tl_detector.py`](https://github.com/nutmas/CapstoneProject-AutoWheels/blob/TensorBranch/ros/src/tl_detector/tl_detector.py)
    - In Parallel to classification thread, the [`tl_detector.py`](https://github.com/nutmas/CapstoneProject-AutoWheels/blob/TensorBranch/ros/src/tl_detector/tl_detector.py)function `run_main()` continuously calculates the nearest traffic light based on current pose, to understand the distance to next stop line. When a position and classification are aligned, the node will only output a waypoint representing distance to stop line, if the traffic light is RED or YELLOW.
    - The [`waypoint_updater.py`](https://github.com/nutmas/CapstoneProject-AutoWheels/blob/TensorBranch/ros/src/waypoint_updater/waypoint_updater.py) receives the stop line waypoint and will control the vehicle to bring it to a stop at the stop line position. Once a green light is present the waypoint is removed and the vehicle accelerates to the set speed.

#### Performance Evaluation<a id="sec-6-2"></a>

- Inception v1 model has lower accuracy but runs faster producing results of ~330ms per classification (On 1050Ti GPU). However this required more classification outputs to establish a confirmed traffic light state.
- Inception v2 model has very high accuracy but runs much slower ~1.5secs per classification (On 1050Ti GPU). This can work on a single state result.
- Both models could successfully navigate the track and obey the traffic lights. However both classifications took over 1 second to have a confirmed state. v1 would sometimes mis-classify a number of times and due to the higher state change requirements could miss a red light.
- The simulator would crash at a certain point sometimes and the styx server crash, this occurred more frequently on the v2 model. Videos showing the performance of each model are shown in the videos:
    + [inception v1 video](https://www.youtube.com/watch?v=G_5z3RUoplA)
    + [inception v2 video](https://www.youtube.com/watch?v=eRHMHTRL228&t=4s)
- I evaluated the models on a 1080Ti GPU which is similar specification to the Udacity hardware. This hardware change significantly improved the speed performance time of the classifiers. The v2 dropped from 1.5s to 650ms and maintained it quality which meant ti was a good solution for successfully navigating the simulator. The results can be seen in this [Video](https://www.youtube.com/watch?v=OaNf-dULUBw)
    
#### Conclusion for end-to-end classifier
The v1 and v2 inception models are similar size once frozen (52MB vs 55MB). However the model which ran for 10x more epoch is significantly slower but has a much higher reliability for classification. The v2 model was chosen as it could perform to the meet the requirement of the simulator track.
No real world data training or testing was performed on the classifier yet; it was therefore judged by the team that the YOLO classifier with Darknet would be more suitable for the submission.
To take this end-to-end classifier forwards it would need retraining on the real world data and have a switch in the launch file to select real world or simulator world models.

#### PID Tuning Parameters<a id="sec-6-3"></a>

**The final values of PID controller for end-to-end Tensorflow model were the following: (KP = 0.25, KI = 0.0, KD = 0.15, MN = 0.0, MX = 0.5).**


## Results<a id="sec-7"></a>

This repo shows the Tensorflow traffic light classifier implementation. This has model not been trained for real world traffic lights; It will successfully navigate the simulator track using the RCNN Inception Net as the end-to-end traffic light Classifier.  
The integration of the classifiers with the control is aligned so that the classifiers can be exchanged with minimal code modification. The actual submission for Real World evaluation with YOLO classifier is submitted by Team Lead Diogo Silva

**This [Video](https://www.youtube.com/watch?v=HVy0eSQZLXA) shows the end-to-end net in operation while the vehicle navigates around the simulator track.**


## License<a id="sec-8"></a>

For License information please see the [LICENSE](./LICENSE) file for details

---

