#ifndef SIM_H
#define SIM_H

#include "control.hpp"
#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include <string>
#include "ros_interface.hpp"
#include "ros/ros.h"

class Simulation {
public:
  mjModel *m = nullptr;
  mjData *d = nullptr;
  mjvScene scn;
  mjvCamera cam;
  mjvOption opt;
  mjrContext con;
  GLFWwindow *window = nullptr;

  mjvCamera sensorCam;
  ros::Publisher imagePub;
  bool cameraEnabled = false;
  int camW = 640, camH = 480;
  double camPeriod = 0.1; // period between published images

  Simulation();
  ~Simulation();

  bool initialize(const char *robotModelPath);
  bool enableCamera(ros::NodeHandle &nh, const std::string &camName, int width,
                    int height, double rateHz);
  void publishCamera();
  void run(RobotController &ctrl, RosInterface &rosInterface);
};

#endif
