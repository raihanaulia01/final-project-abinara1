#include "control.hpp"
#include <cstdio>
#include <cstring>
#include <iostream>

RobotController::RobotController(const mjModel *m) : m(m) {
  gripperActuator = mj_name2id(m, mjOBJ_ACTUATOR, "gripper");
  toolSite = mj_name2id(m, mjOBJ_SITE, "tool_tip");

  for (int b = 0; b < m->nbody; b++) {
    const char *name = mj_id2name(m, mjOBJ_BODY, b);
    if (name && std::strncmp(name, "block_", 6) == 0 && m->body_jntnum[b] == 1 &&
        m->jnt_type[m->body_jntadr[b]] == mjJNT_FREE)
      blocks.push_back(b);
  }

  if (gripperActuator < 0)
    fprintf(stderr, "[control] no actuator named 'gripper': gripper commands are ignored\n");
  if (toolSite < 0)
    fprintf(stderr, "[control] no site named 'tool_tip': fake grasp is disabled\n");
  if (toolSite >= 0 && blocks.empty())
    fprintf(stderr, "[control] no free-joint bodies named 'block_*' found\n");
}

void RobotController::applyJointPos(mjData *d) const {
  for (int i = 0; i < jointCount; i++) {
    d->ctrl[i] = jointPos[i];
  }
  if (gripperActuator >= 0) {
    // gripper actuator's ctrlrange upper bound = closed, lower bound = open
    const mjtNum *range = m->actuator_ctrlrange + 2 * gripperActuator;
    d->ctrl[gripperActuator] = gripperClosed ? range[1] : range[0];
  }
}

void RobotController::checkJointLimits() {
  // check joint limits, stored in i*2 for min and i*2 + 1 for max
  for (int i = 0; i < jointCount; i++) {
    if (!m->jnt_limited[i])
      continue;

    double jointMin = m->jnt_range[i*2];
    double jointMax = m->jnt_range[i*2 + 1];

    if (jointPos[i] > jointMax)
      jointPos[i] = jointMax;
    else if (jointPos[i] < jointMin)
      jointPos[i] = jointMin;
  }
}

// Fake grasp: while the gripper is closed and a block was within graspRadius of the tool tip,
// the block is carried rigidly with the tool. Good enough to test the whole pipeline without
// tuning contact friction.
void RobotController::updateGrasp(mjData *d) {
  if (toolSite < 0)
    return;
  const mjtNum *toolPos = d->site_xpos + 3 * toolSite;
  const mjtNum *toolMat = d->site_xmat + 9 * toolSite;

  if (gripperClosed && heldBody < 0) {
    double best = graspRadius;
    int bestBody = -1;
    for (int b : blocks) {
      double dist = mju_dist3(toolPos, d->xpos + 3 * b);
      if (dist < best) {
        best = dist;
        bestBody = b;
      }
    }
    if (bestBody >= 0) {
      heldBody = bestBody;
      mjtNum diff[3];
      mju_sub3(diff, d->xpos + 3 * bestBody, toolPos);
      // position in tool frame
      mju_mulMatTVec(relPos, toolMat, diff, 3, 3); 
      // orientation in tool frame
      mju_mulMatTMat(relMat, toolMat, d->xmat + 9 * bestBody, 3, 3, 3);
    }
  } else if (!gripperClosed && heldBody >= 0) {
    int dof = m->jnt_dofadr[m->body_jntadr[heldBody]];
    mju_zero(d->qvel + dof, 6); // let go: the block drops instead of being flung
    heldBody = -1;
  }

  if (heldBody >= 0) {
    int joint = m->body_jntadr[heldBody];
    int qadr = m->jnt_qposadr[joint];
    int dof = m->jnt_dofadr[joint];

    mjtNum pos[3], mat[9], quat[4];
    mju_mulMatVec(pos, toolMat, relPos, 3, 3);
    mju_addTo3(pos, toolPos);
    mju_mulMatMat(mat, toolMat, relMat, 3, 3, 3);
    mju_mat2Quat(quat, mat);

    mju_copy3(d->qpos + qadr, pos);
    mju_copy4(d->qpos + qadr + 3, quat);
    mju_zero(d->qvel + dof, 6);
  }
}

void RobotController::printJointPos() {
  for (int i = 0; i < jointCount; i++) {
    std::cout << "Joint " << i << ": " << jointPos[i] << " rad\n";
  }
  std::cout << std::endl;
}
