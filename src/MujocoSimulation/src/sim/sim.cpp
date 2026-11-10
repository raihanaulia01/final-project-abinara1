#include "sim.hpp"
#include "sensor_msgs/Image.h"
#include <cstdio>
#include <cstring>
#include <vector>

Simulation::Simulation() {
  mjv_defaultCamera(&cam);
  mjv_defaultCamera(&sensorCam);
  mjv_defaultOption(&opt);
  mjv_defaultScene(&scn);
  mjr_defaultContext(&con);
}

Simulation::~Simulation() {
  mjv_freeScene(&scn);
  mjr_freeContext(&con);
  mj_deleteData(d);
  mj_deleteModel(m);
  glfwTerminate();
}

bool Simulation::initialize(const char *robotModelPath) {
  char error[1000] = "Couldn't load model";
  m = mj_loadXML(robotModelPath, nullptr, error, sizeof(error));
  if (!m) {
    fprintf(stderr, "%s\n", error);
    return false;
  }
  d = mj_makeData(m);

  if (!glfwInit()) {
    fprintf(stderr, "glfwInit() failed\n");
    return false;
  }

  window = glfwCreateWindow(1600, 900, "Arm sim", nullptr, nullptr);
  if (!window) {
    fprintf(stderr, "glfwCreateWindow() failed\n");
    return false;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  mjv_makeScene(m, &scn, 6700);
  mjr_makeContext(m, &con, mjFONTSCALE_100);
  return true;
}

// renders the MJCF camera 'camName' offscreen and publishes it as
// sensor_msgs/Image on /camera/image_raw at rateHz.
bool Simulation::enableCamera(ros::NodeHandle &nh, const std::string &camName,
                              int width, int height, double rateHz) {
  int id = mj_name2id(m, mjOBJ_CAMERA, camName.c_str());
  if (id < 0) {
    fprintf(stderr, "camera '%s' not found in the model\n", camName.c_str());
    return false;
  }
  if (width > m->vis.global.offwidth || height > m->vis.global.offheight) {
    fprintf(stderr,
            "camera %dx%d is larger than the offscreen buffer %dx%d; set "
            "<visual><global offwidth=\"..\" offheight=\"..\"/></visual> in "
            "the XML\n",
            width, height, m->vis.global.offwidth, m->vis.global.offheight);
    return false;
  }

  mjv_defaultCamera(&sensorCam);
  sensorCam.type = mjCAMERA_FIXED;
  sensorCam.fixedcamid = id;
  camW = width;
  camH = height;
  camPeriod = rateHz > 0 ? 1.0 / rateHz : 0.1;
  imagePub = nh.advertise<sensor_msgs::Image>("/camera/image_raw", 1);
  cameraEnabled = true;
  return true;
}

void Simulation::publishCamera() {
  mjrRect rect = {0, 0, camW, camH};
  mjr_setBuffer(mjFB_OFFSCREEN, &con);
  mjv_updateScene(m, d, &opt, nullptr, &sensorCam, mjCAT_ALL, &scn);
  mjr_render(rect, &scn, &con);

  std::vector<unsigned char> rgb(3 * camW * camH);
  mjr_readPixels(rgb.data(), nullptr, rect, &con);
  mjr_setBuffer(mjFB_WINDOW, &con);

  sensor_msgs::Image img;
  img.header.stamp = ros::Time::now();
  img.header.frame_id = "arena_cam";
  img.width = camW;
  img.height = camH;
  img.encoding = "rgb8";
  img.step = 3 * camW;
  img.data.resize(3 * camW * camH);
  for (int r = 0; r < camH; r++) // OpenGL rows are bottom-up. flip vertically
    std::memcpy(&img.data[r * 3 * camW], &rgb[(camH - 1 - r) * 3 * camW],
                3 * camW);
  imagePub.publish(img);
}

void Simulation::run(RobotController &ctrl, RosInterface &rosInterface) {
  const double wallStart = glfwGetTime();
  double nextImage = 0.0;

  while (!glfwWindowShouldClose(window) && ros::ok()) {
    ros::spinOnce();

    // advance physics until simulated time catches up with the clock
    int steps = 0;
    while (d->time < glfwGetTime() - wallStart && steps++ < 50) {
      ctrl.applyJointPos(d);
      ctrl.updateGrasp(d);
      mj_step(m, d);
    }

    rosInterface.publishJointPos(d);

    if (cameraEnabled && glfwGetTime() >= nextImage) {
      publishCamera();
      nextImage = glfwGetTime() + camPeriod;
    }

    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
}
