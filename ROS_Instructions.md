# ROS instructions

Clone this fork into your ROS workspace in the /src folder (with your other packages). Additionally, to receive images from your webcam, use the the ROS [usb_cam](http://wiki.ros.org/usb_cam "ROS wiki") package. After cloning both these packages into your /src folder, run ```catkin_make``` in the root directory of your ROS workspace. 

Additionally, 
1. Run ```roscore``` to initialize ROS
2. Run ```roslaunch usb_cam usb_cam-test.launch``` to run the usb_cam node
3. Next, to run object inference, navigate to the Detectron package and run the command```rosrun detectron infer_simple.py .```  This will cause images with object segmentation and labeling to be published on a /detectron_output topic through an Image ROS message type. 
4. To view published images, run ```rosrun image_view image_view image:=/detectron_output``` in a seperate terminal. 



todo: insert finetuning instructions here


# Troubleshooting
* Error message = "AttributeError: get_image instance has no attribute 'astype'" <br/>
Possible fix: ensure that the usb_cam package has been launched

* Error message "usage: infer_simple.py...." <br/>
Possible fix: ensure that there is a period ``` . ``` at the end of the ``` rosrun ``` command.

