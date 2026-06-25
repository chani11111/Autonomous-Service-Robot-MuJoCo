"""
simulation_manager.py — v5 restored + two fixes:
  1. get_robot_position reads from qpos (correct for slide joints)
  2. get_robot_yaw reads from hinge_z qpos (correct)
Everything else is identical to the version where the robot
moved stably and reached the target at step 1063.
"""

import mujoco
import mujoco.viewer
import numpy as np


def build_scene(target_x: float = 3.0, target_y: float = 0.0) -> str:
    return f"""
<mujoco model="office_robot_v5">

  <option timestep="0.01" gravity="0 0 -9.81"/>

  <worldbody>

    <light diffuse=".65 .65 .60" pos="0 0 5" dir="0 0 -1" castshadow="false"/>
    <light diffuse=".25 .25 .28" pos="4 4 3" dir="-1 -1 -1" castshadow="false"/>

    <geom name="floor" type="plane" size="7 7 0.1"
          rgba=".78 .76 .72 1" friction="1.0 0.05 0.05"/>

    <!-- Desks -->
    <geom name="d1_top" type="box" size=".55 .38 .04" pos=" 2.8  2.8  .74" rgba=".55 .42 .28 1"/>
    <geom name="d1_l1"  type="cylinder" size=".04 .35" pos=" 2.34 2.50  .35" rgba=".35 .25 .15 1"/>
    <geom name="d1_l2"  type="cylinder" size=".04 .35" pos=" 3.26 2.50  .35" rgba=".35 .25 .15 1"/>
    <geom name="d1_l3"  type="cylinder" size=".04 .35" pos=" 2.34 3.10  .35" rgba=".35 .25 .15 1"/>
    <geom name="d1_l4"  type="cylinder" size=".04 .35" pos=" 3.26 3.10  .35" rgba=".35 .25 .15 1"/>

    <geom name="d2_top" type="box" size=".55 .38 .04" pos="-2.8  2.8  .74" rgba=".55 .42 .28 1"/>
    <geom name="d2_l1"  type="cylinder" size=".04 .35" pos="-3.26 2.50  .35" rgba=".35 .25 .15 1"/>
    <geom name="d2_l2"  type="cylinder" size=".04 .35" pos="-2.34 2.50  .35" rgba=".35 .25 .15 1"/>
    <geom name="d2_l3"  type="cylinder" size=".04 .35" pos="-3.26 3.10  .35" rgba=".35 .25 .15 1"/>
    <geom name="d2_l4"  type="cylinder" size=".04 .35" pos="-2.34 3.10  .35" rgba=".35 .25 .15 1"/>

    <geom name="d3_top" type="box" size=".55 .38 .04" pos=" 2.8 -2.8  .74" rgba=".55 .42 .28 1"/>
    <geom name="d3_l1"  type="cylinder" size=".04 .35" pos=" 2.34 -3.10 .35" rgba=".35 .25 .15 1"/>
    <geom name="d3_l2"  type="cylinder" size=".04 .35" pos=" 3.26 -3.10 .35" rgba=".35 .25 .15 1"/>
    <geom name="d3_l3"  type="cylinder" size=".04 .35" pos=" 2.34 -2.50 .35" rgba=".35 .25 .15 1"/>
    <geom name="d3_l4"  type="cylinder" size=".04 .35" pos=" 3.26 -2.50 .35" rgba=".35 .25 .15 1"/>

    <geom name="d4_top" type="box" size=".55 .38 .04" pos="-2.8 -2.8  .74" rgba=".55 .42 .28 1"/>
    <geom name="d4_l1"  type="cylinder" size=".04 .35" pos="-3.26 -3.10 .35" rgba=".35 .25 .15 1"/>
    <geom name="d4_l2"  type="cylinder" size=".04 .35" pos="-2.34 -3.10 .35" rgba=".35 .25 .15 1"/>
    <geom name="d4_l3"  type="cylinder" size=".04 .35" pos="-3.26 -2.50 .35" rgba=".35 .25 .15 1"/>
    <geom name="d4_l4"  type="cylinder" size=".04 .35" pos="-2.34 -2.50 .35" rgba=".35 .25 .15 1"/>

    <!-- Target marker -->
    <geom name="target_pole" type="cylinder" size=".03 .3"
          pos="{target_x} {target_y} .3"
          rgba=".9 .1 .1 1" contype="0" conaffinity="0"/>
    <geom name="target_top"  type="sphere"   size=".07"
          pos="{target_x} {target_y} .65"
          rgba=".9 .1 .1 1" contype="0" conaffinity="0"/>
    <geom name="target_ring" type="cylinder" size=".15 .01"
          pos="{target_x} {target_y} .01"
          rgba=".9 .1 .1 .4" contype="0" conaffinity="0"/>

    <!-- Dynamic obstacles -->
    <body name="dynamic1" pos="1.5 0 .2">
      <freejoint name="dyn1_joint"/>
      <geom type="sphere" size=".18" rgba=".85 .55 .1 1" mass="1.0"/>
    </body>
    <body name="dynamic2" pos="-1.5 0 .2">
      <freejoint name="dyn2_joint"/>
      <geom type="sphere" size=".18" rgba=".85 .55 .1 1" mass="1.0"/>
    </body>

    <!-- Robot — slide+hinge (stable, no pitching) -->
    <body name="robot_base" pos="0 0 0.001">
      <joint name="slide_x" type="slide" axis="1 0 0" limited="false"/>
      <joint name="slide_y" type="slide" axis="0 1 0" limited="false"/>
      <joint name="hinge_z" type="hinge" axis="0 0 1" limited="false"/>

      <body name="robot" pos="0 0 0.052">

        <geom name="ballast" type="cylinder"
              size=".17 .012" pos="0 0 -.025"
              rgba=".08 .08 .08 1" mass="2.5"/>

        <geom name="body_main" type="cylinder"
              size=".19 .038" rgba=".18 .18 .18 1" mass="0.8"/>

        <geom name="body_dome" type="cylinder"
              size=".11 .022" pos="0 0 .052"
              rgba=".25 .25 .25 1" mass="0.1"/>

        <geom name="sensor_front" type="sphere" size=".022"
              pos=".17 0 .025" rgba=".9 .4 .1 1" mass="0.01"/>

        <geom name="status_light" type="sphere" size=".016"
              pos=".07 0 .08" rgba=".2 .5 1.0 1" mass="0.01"/>

        <site name="rf_front"       pos=" .19  .00 .03" euler="0 0   0"/>
        <site name="rf_front_left"  pos=" .13  .13 .03" euler="0 0  45"/>
        <site name="rf_left"        pos=" .00  .19 .03" euler="0 0  90"/>
        <site name="rf_rear_left"   pos="-.13  .13 .03" euler="0 0 135"/>
        <site name="rf_rear"        pos="-.19  .00 .03" euler="0 0 180"/>
        <site name="rf_rear_right"  pos="-.13 -.13 .03" euler="0 0 225"/>
        <site name="rf_right"       pos=" .00 -.19 .03" euler="0 0 270"/>
        <site name="rf_front_right" pos=" .13 -.13 .03" euler="0 0 315"/>

        <body name="wheel_left" pos="0 .148 -.018">
          <joint name="joint_left" type="hinge"
                 axis="0 1 0" limited="false" damping="0.3"/>
          <geom type="cylinder" size=".048 .020" euler="90 0 0"
                rgba=".08 .08 .08 1" mass="0.2"
                friction="1.5 0.05 0.05"/>
        </body>

        <body name="wheel_right" pos="0 -.148 -.018">
          <joint name="joint_right" type="hinge"
                 axis="0 1 0" limited="false" damping="0.3"/>
          <geom type="cylinder" size=".048 .020" euler="90 0 0"
                rgba=".08 .08 .08 1" mass="0.2"
                friction="1.5 0.05 0.05"/>
        </body>

      </body>
    </body>

  </worldbody>

  <actuator>
    <velocity name="motor_left"  joint="joint_left"
              kv="20" forcelimited="true" forcerange="-5 5"/>
    <velocity name="motor_right" joint="joint_right"
              kv="20" forcelimited="true" forcerange="-5 5"/>
  </actuator>

  <sensor>
    <rangefinder name="rf_front"       site="rf_front"       cutoff="4.0"/>
    <rangefinder name="rf_front_left"  site="rf_front_left"  cutoff="4.0"/>
    <rangefinder name="rf_left"        site="rf_left"        cutoff="4.0"/>
    <rangefinder name="rf_rear_left"   site="rf_rear_left"   cutoff="4.0"/>
    <rangefinder name="rf_rear"        site="rf_rear"        cutoff="4.0"/>
    <rangefinder name="rf_rear_right"  site="rf_rear_right"  cutoff="4.0"/>
    <rangefinder name="rf_right"       site="rf_right"       cutoff="4.0"/>
    <rangefinder name="rf_front_right" site="rf_front_right" cutoff="4.0"/>
  </sensor>

</mujoco>
"""


class SimulationManager:
    """S — Single Responsibility: MuJoCo interface only."""

    SENSOR_NAMES = [
        "rf_front", "rf_front_left", "rf_left", "rf_rear_left",
        "rf_rear", "rf_rear_right", "rf_right", "rf_front_right"
    ]

    def __init__(self, target_x: float = 3.0, target_y: float = 0.0):
        self.model  = mujoco.MjModel.from_xml_string(
            build_scene(target_x, target_y)
        )
        self.data   = mujoco.MjData(self.model)
        self.viewer = None

        self._robot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "robot"
        )

        # qpos indices for the planar joints
        def jidx(name):
            return self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        self._jx = jidx("slide_x")
        self._jy = jidx("slide_y")
        self._jz = jidx("hinge_z")
    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)

    def sync_viewer(self) -> None:
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()

    def is_running(self) -> bool:
        return self.viewer is not None and self.viewer.is_running()

    def get_timestep(self) -> float:
        return self.model.opt.timestep

    def get_robot_position(self) -> np.ndarray:
        # Read directly from slide joint qpos — correct for this joint type
        x = float(self.data.qpos[self._jx])
        y = float(self.data.qpos[self._jy])
        return np.array([x, y, 0.052])

    def get_robot_yaw(self) -> float:
        # Read directly from hinge_z qpos
        return float(np.degrees(self.data.qpos[self._jz]))

    def get_robot_pose(self) -> tuple:
        pos = self.get_robot_position()
        return pos[0], pos[1], self.get_robot_yaw()

    # Ray directions in robot-local frame (8 directions, every 45°)
    _RAY_DIRS = np.array([
        [ 1.0,    0.0,   0.0],
        [ 0.707,  0.707, 0.0],
        [ 0.0,    1.0,   0.0],
        [-0.707,  0.707, 0.0],
        [-1.0,    0.0,   0.0],
        [-0.707, -0.707, 0.0],
        [ 0.0,   -1.0,   0.0],
        [ 0.707, -0.707, 0.0],
    ])

    def get_raw_sensor_data(self) -> np.ndarray:
        """
        Cast rays using mj_ray — works correctly with slide+hinge joints.
        Returns distance in meters, or -1.0 if nothing within 4m.
        """
        mujoco.mj_forward(self.model, self.data)

        yaw = self.data.qpos[self._jz]
        c, s = np.cos(yaw), np.sin(yaw)
        rot  = np.array([[c, -s, 0.0],
                         [s,  c, 0.0],
                         [0,  0, 1.0]])

        readings = np.full(8, -1.0)
        geom_id  = np.array([-1], dtype=np.int32)

        for i in range(8):
            site_id  = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, self.SENSOR_NAMES[i]
            )
            pos       = self.data.site_xpos[site_id].copy()
            world_dir = rot @ self._RAY_DIRS[i]

            dist = mujoco.mj_ray(
                self.model, self.data,
                pos, world_dir,
                None, 1,
                self._robot_id,
                geom_id
            )
            if 0.0 < dist <= 4.0:
                readings[i] = dist

        return readings

    def set_wheel_velocities(self, left: float, right: float) -> None:
        self.data.ctrl[0] = left
        self.data.ctrl[1] = right

    def move_dynamic_objects(self, time: float) -> None:
        def qidx(name):
            return self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]

        i1 = qidx("dyn1_joint")
        self.data.qpos[i1]     = 1.8 * np.cos(time * 0.5)
        self.data.qpos[i1 + 1] = 1.8 * np.sin(time * 0.5)

        i2 = qidx("dyn2_joint")
        self.data.qpos[i2]     = -1.8
        self.data.qpos[i2 + 1] = 2.2 * np.sin(time * 0.7)

    def __enter__(self):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self

    def __exit__(self, *args):
        pass