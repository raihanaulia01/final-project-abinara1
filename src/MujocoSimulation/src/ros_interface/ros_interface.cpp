#include "ros_interface.hpp"
#include <string>

RosInterface::RosInterface(ros::NodeHandle &nodeHandle,
                           RobotController &controller)
    : controller(controller) {
  commandSubscriber = nodeHandle.subscribe(
      "/joint_command", 100, &RosInterface::commandCallback, this);
  gripperSubscriber = nodeHandle.subscribe(
      "/gripper_command", 10, &RosInterface::gripperCallback, this);

  jointStatePublisher =
      nodeHandle.advertise<sensor_msgs::JointState>("/joint_states", 100);
}

void RosInterface::commandCallback(
    const sensor_msgs::JointState::ConstPtr &message) {
  if (message->position.size() != jointCount) {
    ROS_WARN("Expected %d joint positions, received %zu instead.", jointCount,
             message->position.size());
    return;
  }

  double pos[jointCount];
  for (int i = 0; i < jointCount; i++)
    pos[i] = message->position[i];

  // if names are given, order by name (so a message with a different order still works)
  if (!message->name.empty()) {
    if (message->name.size() != message->position.size()) {
      ROS_WARN("joint_command: %zu names but %zu positions.", message->name.size(),
               message->position.size());
      return;
    }
    for (int i = 0; i < jointCount; i++) {
      bool found = false;
      for (size_t k = 0; k < message->name.size(); k++) {
        if (message->name[k] == jointNames[i]) {
          pos[i] = message->position[k];
          found = true;
          break;
        }
      }
      if (!found) {
        ROS_WARN("joint_command is missing joint '%s'.", jointNames[i]);
        return;
      }
    }
  }

  for (int i = 0; i < jointCount; i++) {
    controller.jointPos[i] = pos[i];
  }

  controller.checkJointLimits();
}

void RosInterface::gripperCallback(const std_msgs::String::ConstPtr &message) {
  if (message->data == "close")
    controller.gripperClosed = true;
  else if (message->data == "open")
    controller.gripperClosed = false;
  else
    ROS_WARN("gripper_command must be 'open' or 'close', got '%s'.",
             message->data.c_str());
}

void RosInterface::publishJointPos(const mjData *d) {
  sensor_msgs::JointState message;
  message.header.stamp = ros::Time::now();
  for (int i = 0; i < jointCount; i++) {
    message.name.push_back(jointNames[i]);
    message.position.push_back(d->qpos[i]);
  }

  jointStatePublisher.publish(message);
}
