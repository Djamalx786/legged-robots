import pybullet as pb
import numpy as np
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot
import pinocchio as pin

# For REEM-C robot
#urdf = "src/reemc_description/robots/reemc.urdf"
#path_meshes = "src/reemc_description/meshes/../.."

# For Talos robot
urdf = "src/talos_description/robots/talos_reduced.urdf"
path_meshes = "src/talos_description/meshes/../.."

'''
Talos
0, 1, 2, 3, 4, 5, 			    # left leg
6, 7, 8, 9, 10, 11, 			# right leg
12, 13,                         # torso
14, 15, 16, 17, 18, 19, 20, 21  # left arm
22, 23, 24, 25, 26, 27, 28, 29  # right arm
30, 31                          # head

REEMC
0, 1, 2, 3, 4, 5, 			    # left leg
6, 7, 8, 9, 10, 11, 			# right leg
12, 13,                         # torso
14, 15, 16, 17, 18, 19, 20,     # left arm
21, 22, 23, 24, 25, 26, 27,     # right arm
28, 29                          # head
'''

# Initial condition for the simulator an model
z_init = 1.15
q_actuated_home = np.zeros(32)
q_actuated_home[:6] = np.array([0, 0, -0.44, 0.9, -0.45, 0])
q_actuated_home[6:12] = np.array([0, 0, -0.44, 0.9, -0.45, 0])
q_actuated_home[14:22] = np.array([0, -0.24, 0, -1, 0, 0, 0, 0 ])
q_actuated_home[22:30] = np.array([0, -0.24, 0, -1, 0,  0, 0, 0 ])

# Initialization position including floating base
q_home = np.hstack([np.array([0, 0, z_init, 0, 0, 0, 1]), q_actuated_home])

# setup the task stack
modelWrap = pin.RobotWrapper.BuildFromURDF(urdf,                        # Model description
                                           path_meshes,                 # Model geometry descriptors 
                                           pin.JointModelFreeFlyer(),   # Floating base model. Use "None" if fixed
                                           True,                        # Printout model details
                                           None)                        # Load meshes different from the descripor
# Get model from wrapper
model = modelWrap.model

# setup the simulator
simulator = PybulletWrapper(sim_rate=1000)

#Create Pybullet-Pinocchio map
robot = Robot(simulator,            # The Pybullet wrapper
              urdf,                 # Robot descriptor
              model,                # Pinocchio model
              [0, 0, z_init],       # Floating base initial position
              [0,0,0,1],            # Floating base initial orientation [x,y,z,w]
              q=q_home,             # Initial state
              useFixedBase=False,   # Fixed base or not
              verbose=True)         # Printout details
data = robot._model.createData()

M = pin.ccrba(robot._model, data, robot._q, robot._v)  # Compute the mass matrix at home position
#print("Mass-Inertia matrix at home position:\n", M)

b = pin.nonLinearEffects(robot._model , data, robot._q , robot._v)  # Compute the non-linear effects at home position
#print("Non-linear effects at home position:\n", b)
#Needed for compatibility
simulator.addLinkDebugFrame(-1,-1)


# Setup pybullet camera
pb.resetDebugVisualizerCamera(
    cameraDistance=1.2,
    cameraYaw=90,
    cameraPitch=-20,
    cameraTargetPosition=[0.0, 0.0, 0.8])

# Joint command vector
tau = q_actuated_home*0


def spline_interpolation(model, q_init, q_home, duration, sim_rate): 
        steps = int(duration * sim_rate)  # Calculation of steps
        trajectory = []
        print(f"Number of steps for spline interpolation: {steps}")
        for i in range(steps + 1):
            t = min(1.0, i / steps)
            q_i = pin.interpolate(model, q_init, q_home, t)
            trajectory.append(q_i.copy())
        
        return trajectory


# Generate spline trajectory
duration = 10  # seconds
print(f"simulator rate: {simulator._sim_rate}")

trajectory = spline_interpolation(robot._model, robot._q, q_home, duration, simulator._sim_rate)
print(f"Diese Groesse will ich : ", np.shape(robot._model))
# Main simulation loop
done = False
step = 0

while not done:
    # update the simulator and the robot
    simulator.step()
    simulator.debug()
    robot.update()

    # Feed spline trajectory as desired joint positions
    if step < len(trajectory):
        q_d = trajectory[step]
    else:
        q_d = q_home
        done = True  # Exit the loop when the trajectory is complete
        break

    # Joint space controller
    K_p = np.zeros(32)
    K_d = np.zeros(32)
    K_p[0:12] = 1000  # Increase gains for legs
    K_p[12:14] = 540  # Increase gains for torso
    K_p[14:] = 50  # Increase gains for arms
    K_d[0:12] = 13  # Increase derivative gains for legs
    K_d[12:14] = 5  # Increase derivative gains for torso
    K_d[14:] = 2  # Increase derivative gains for arms
    tau = K_p * (q_d[7:39] - robot._q[7:39]) + K_d * (-robot._v[6:38])  # PD control law

    robot.setActuatedJointTorques(tau)

    step += 1  # Increment the step counter

