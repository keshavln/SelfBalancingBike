import time
import mujoco
import mujoco.viewer
import math
import pynput
import random
from pynput.keyboard import Key, Listener

#---------CONSTANTS---------
KP = 7
KD = 2.5
VARIATION = 20.0
EPISODE_LENGTH = 90
#---------------------------

m = mujoco.MjModel.from_xml_path('model.xml')
d = mujoco.MjData(m)

# Setting initial tilt

random_angle_deg = random.uniform(-VARIATION, VARIATION)
random_angle_rad = math.radians(random_angle_deg)

joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "root")
qpos_adr = m.jnt_qposadr[joint_id]

d.qpos[qpos_adr + 3] = math.cos(random_angle_rad / 2.0)
d.qpos[qpos_adr + 4] = 0.0
d.qpos[qpos_adr + 5] = math.sin(random_angle_rad / 2.0)
d.qpos[qpos_adr + 6] = 0.0

mujoco.mj_forward(m, d)

# Retrieve sensor and motor IDs

orientation_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "chassis_orientation")
gyro_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "chassis_gyro")

orientation_adr = m.sensor_adr[orientation_id]
gyro_adr = m.sensor_adr[gyro_id]

reaction_motor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "reaction_wheel_motor")
front_motor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_wheel_motor")
steering_motor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "steering_motor")
back_motor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "back_wheel_motor")

# Global variables controlling speed and steering

forward_speed = 0
throttle = False
steer = 0

def quat_to_rpy(quat):
  w,x,y,z = quat
  roll = math.atan2(2*(w*x+y*z), 1-2*(x**2+y**2))
  pitch = math.asin(2*(w*y-z*x))
  yaw = math.atan2(2*(w*z+x*y), 1-2*(y**2+z**2))
  return roll, pitch, yaw

def on_press(key):
  global forward_speed, steer, throttle
  if key == Key.up:
    throttle = True
    forward_speed += -0.02
  elif key == Key.down:
    throttle = True
    forward_speed += 0.02
  elif key == Key.left:
    steer = 0.01
  elif key == Key.right:
    steer = -0.01

def on_release(key):
  global forward_speed, steer, throttle
  if key in [Key.up, Key.down]:
    throttle = False
  elif key in [Key.left, Key.right]:
    steer = 0

listener = Listener(on_press=on_press, on_release=on_release)
listener.start()

with mujoco.viewer.launch_passive(m, d) as viewer:

  #chassis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chassis")
  cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "bike_cam")
  viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
  viewer.cam.fixedcamid = cam_id

  #viewer.cam.distance = 0.2     # How far away the camera is (zoom)
  #viewer.cam.azimuth = 200         # Angle around the Z-axis (in degrees)
  #viewer.cam.elevation = -30      # Camera tilt (pitch) down toward the bike
  #viewer.cam.lookat[:] = [0, 0.05, 0.1]

  start = time.time()
  while viewer.is_running() and time.time() - start < EPISODE_LENGTH:
    step_start = time.time()

    d.ctrl[front_motor] = forward_speed
    d.ctrl[back_motor] = forward_speed
    d.ctrl[steering_motor] = steer

    rpy = quat_to_rpy(d.sensordata[orientation_adr : orientation_adr + 4])
    wrpy = d.sensordata[gyro_adr : gyro_adr + 3]
    pitch = rpy[1]
    pitch_rate = wrpy[1]

    # PID Control
    acceleration = KP*pitch + KD*pitch_rate

    d.ctrl[reaction_motor] += m.opt.timestep * acceleration

    mujoco.mj_step(m, d)
    print("speed", forward_speed)

    if not throttle:
      forward_speed += -forward_speed * 0.001

    viewer.sync()

    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)