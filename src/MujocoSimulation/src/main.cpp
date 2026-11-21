#include "ros/ros.h"
#include "callbacks.hpp"
#include "control.hpp"
#include "sim.hpp"
#include "ros_interface.hpp"

int main(int argc, char **argv) {
  ros::init(argc, argv, "mujoco_arm_sim");
  ros::NodeHandle nh("~");

  std::string model_path;
  if (!nh.getParam("model_path", model_path)) {
    ROS_FATAL("Parameter model_path is missing.");
    return 1;
  }

  glfwSetErrorCallback(callbacks::glfw_error_callback);
  Simulation sim;

  if (!sim.initialize(model_path.c_str()))
    return 1;

  RobotController ctrl(sim.m, nh);
  RosInterface rosInterface(nh, ctrl);
  callbacks::initialize(&sim, &ctrl, sim.window);

  bool cameraEnabled = true;
  nh.param("camera_enabled", cameraEnabled, true);
  if (cameraEnabled) {
    std::string camName;
    int camWidth, camHeight;
    double camRate;
    nh.param<std::string>("camera_name", camName, "arena_cam");
    nh.param("camera_width", camWidth, 640);
    nh.param("camera_height", camHeight, 480);
    nh.param("camera_rate", camRate, 10.0);
    if (!sim.enableCamera(nh, camName, camWidth, camHeight, camRate))
      ROS_WARN("Simulated camera disabled.");
  }

  sim.run(ctrl, rosInterface);
  return 0;
}
