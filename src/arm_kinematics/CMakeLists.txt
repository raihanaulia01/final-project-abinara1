cmake_minimum_required(VERSION 3.0.2)
project(arm_kinematics)

find_package(catkin REQUIRED COMPONENTS
  rospy
  arm_msgs
)

catkin_python_setup()
catkin_package()

catkin_install_python(PROGRAMS
  scripts/ik_node.py
  scripts/check_reach.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
