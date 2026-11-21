cmake_minimum_required(VERSION 3.0.2)
project(arm_hardware)
find_package(catkin REQUIRED COMPONENTS rospy sensor_msgs std_msgs arm_kinematics)
catkin_python_setup()
catkin_package()
catkin_install_python(PROGRAMS
  scripts/bridge_node.py
  scripts/fake_mcu.py
  scripts/presence_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
