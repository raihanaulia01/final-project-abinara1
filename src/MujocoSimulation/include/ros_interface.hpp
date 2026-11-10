#ifndef ROS_INTERFACE_H
#define ROS_INTERFACE_H

#include "control.hpp"
#include "ros/ros.h"
#include "sensor_msgs/JointState.h"
#include "std_msgs/String.h"

class RosInterface {
private:
  void commandCallback(const sensor_msgs::JointState::ConstPtr &message);
  void gripperCallback(const std_msgs::String::ConstPtr &message);
  ros::Subscriber commandSubscriber;
  ros::Subscriber gripperSubscriber;
  ros::Publisher jointStatePublisher;
  RobotController &controller;
  const char *jointNames[jointCount] = {"base", "shoulder", "elbow", "wrist"};

public:
  RosInterface(ros::NodeHandle &nodeHandle, RobotController &controller);
  void publishJointPos(const mjData *d);
};

#endif
