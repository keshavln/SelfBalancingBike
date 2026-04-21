import time
import math
import mujoco
import mujoco.viewer
import random
import control
import numpy as np
from pynput.keyboard import Key, Listener

#---------CONSTANTS---------
VARIATION = 20.0            # random angle initialization range
EPISODE_LENGTH = 90      
m = 0.3788332389620107      # mass
I_b = 0.00061588            # roll moment of inertia of body
I_w = 0.00015627            # reaction wheel moment of inertia
g = 9.81                    # acceleration due to gravity
l = 0.0957917565221756      # height of center of mass
#---------------------------

# Solving the Ricatti equation to obtain the gains
# x = [theta, theta_dot, wheel_speed]

A = np.array([
    [0, 1, 0],
    [(m*g*l)/I_b, 0, 0],
    [-(m*g*l)/I_b, 0, 0]
])

B = np.array([
    [0],
    [-1/I_b],
    [(I_b + I_w) / (I_b * I_w)] 
])

Q = np.diag([
    800.0,
    10.0,
    0.0001
])

R = np.array([[1]])

K, S, E = control.lqr(A, B, Q, R)

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

sensors = {}
for sensor in ['chassis_orientation', 'chassis_gyro', 'rxn_gyro']:
    sensors[sensor] = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor)]

motors = {}
for motor in ['reaction', 'front', 'steering', 'back']:
    motors[motor] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{motor}_motor")

# Global variables controlling speed and steering

forward_speed = 0
throttle = False
steer = 0

# Helper functions

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

  cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "bike_cam")
  viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
  viewer.cam.fixedcamid = cam_id

  start = time.time()

  # Main simulation loop

  while viewer.is_running() and time.time() - start < EPISODE_LENGTH:
    step_start = time.time()

    d.ctrl[motors['front']] = forward_speed
    d.ctrl[motors['back']] = forward_speed
    d.ctrl[motors['steering']] = steer

    # Obtaining current state

    rpy = quat_to_rpy(d.sensordata[sensors['chassis_orientation'] : sensors['chassis_orientation'] + 4])
    wrpy = d.sensordata[sensors['chassis_gyro'] : sensors['chassis_gyro'] + 3]
    rxn_speed = d.sensordata[sensors['rxn_gyro']+1]
    pitch = rpy[1]
    pitch_rate = wrpy[1]

    # LQR control
    
    output = -K[0][0]*pitch - K[0][1]*pitch_rate - K[0][2]*rxn_speed

    d.ctrl[motors['reaction']] += m.opt.timestep * output

    mujoco.mj_step(m, d)

    if not throttle:
      forward_speed += -forward_speed * 0.001

    viewer.sync()

    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
