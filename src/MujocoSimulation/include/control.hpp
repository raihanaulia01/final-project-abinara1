#ifndef CONTROL_H
#define CONTROL_H

#include <mujoco/mujoco.h>
#include <vector>

// Number of arm joints. the gripper isn't counted. Arm must be defined before blocks in the MJCF.
const int jointCount = 4;

class RobotController {
private:
  const mjModel *m;

  // gripper and fake grasp
  int gripperActuator = -1;
  int toolSite = -1;
  std::vector<int> blocks;
  int heldBody = -1;
  double relPos[3] = {0, 0, 0};
  double relMat[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  static constexpr double graspRadius = 0.03;

public:
  double jointPos[jointCount] = {0, 0, 0, 0};
  bool gripperClosed = false;

  RobotController(const mjModel *m);

  void checkJointLimits();
  void applyJointPos(mjData *d) const;
  void updateGrasp(mjData *d);
  void printJointPos();
};

#endif
