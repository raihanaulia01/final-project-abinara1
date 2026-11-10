cmake_minimum_required(VERSION 3.0.2)
project(arm_vision)
find_package(catkin REQUIRED COMPONENTS rospy sensor_msgs cv_bridge arm_msgs)
catkin_python_setup()
catkin_package()
catkin_install_python(PROGRAMS
  scripts/detector_node.py
  scripts/calibrate_homography.py
  scripts/tune_hsv.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
