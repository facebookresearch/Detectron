# ROS instructions

Clone this fork into your ROS workspace in the /src folder (with your other packages). Additionally, to receive images from your webcam, use the the ROS usb_cam package. Clone this into your /src folder, run ```catkin_make``` in the root directory of your ROS workspace. 

1. Run ```roscore``` to initialize ROS
2. Run ```roslaunch usb_cam usb_cam-test.launch``` to run the usb_cam node
3. Next, to run object inference, navigate to the Detectron package and run the command```rosrun detectron infer_simple.py .```  This will cause images with object segmentation and labeling to be published on a /detectron_output topic through an Image ROS message type. 
4. To view published images, run ```rosrun image_view image_view image:=/detectron_output``` in a seperate terminal. 


todo: insert custom dataset instructions here


# Troubleshooting
* "AttributeError: get_image instance has no attribute 'astype'
Possible fix: ensure that the usb_cam package has been launched

* usage: infer_simple.py....
Possible fix: ensure that there is a period ``` . ``` at the end of the ``` rosrun ``` command.

