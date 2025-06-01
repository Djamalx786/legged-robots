import numpy as np
import numpy.linalg as la

# simulator (#TODO: set your own import path!)
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot
from std_msgs.msg import Header 

# modeling
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

from enum import Enum

# ROS
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

################################################################################
# utility functions
################################################################################

class State(Enum):
    JOINT_SPLINE = 0,
    CART_SPLINE = 1

################################################################################
# Robot
################################################################################

class Talos(Robot):
    def __init__(self, simulator, q=None, verbose=True, useFixedBase=True):
        #TODO: Create RobotWrapper (fixed base), Call base class constructor, make publisher  
        # For Talos robot
        self.urdf = "src/talos_description/robots/talos_reduced.urdf"
        self.path_meshes = "src/talos_description/meshes/../.."
        
        self.wrapper = pin.RobotWrapper.BuildFromURDF(self.urdf,                # Model description
                                           self.path_meshes,                 # Model geometry descriptors 
                                           None,                        # Floating base model. Use "None" if fixed
                                           True,                        # Printout model details
                                           None)                        # Load meshes different from the descripor                                            
        print(f"Size of model: {self.wrapper.model.nq}")

        self.model = self.wrapper.model
        self.robot_data = self.wrapper.data
        
        self.initial_height = 1.15  # Initial height of the robot
        # Call base class constructor
        super().__init__(
                    simulator,
                    self.urdf,
                    self.model,
                    [0, 0, self.initial_height],
                    [0, 0, 0, 1],
                    q=q,
                    useFixedBase=True,
                    verbose=True,
                )
        self.node = None
        None
        
    def update(self):
        # TODO: update base class, update pinocchio robot wrapper's kinematics
        super().update()  
        self.wrapper.forwardKinematics(self._q)
        None
    
    def wrapper(self):
        return self._wrapper

    def data(self):
        return self._wrapper.data 
    
    def publish(self):
        # TODO: publish robot state to ros
        if self.joint_state_publisher is None or self.node is None:
            raise RuntimeError("Joint state publisher or node is not initialized.")
        
        q = self._q
        v = self._v 
        tau = np.zeros(len(q))

        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = self.node.get_clock().now().to_msg()
        joint_state_msg.name = self.actuatedJointNames()
        joint_state_msg.position = q.tolist()
        joint_state_msg.velocity = v.tolist()
        joint_state_msg.effort = tau.tolist()
        
        self.joint_state_publisher.publish(joint_state_msg)

################################################################################
# Controllers
################################################################################

class JointSpaceController:
    """JointSpaceController
    Tracking controller in jointspace
    """
    def __init__(self, robot, Kp, Kd):        
        # Save gains, robot ref
        self.robot = robot
        self.Kp = np.array(Kp)  # Proportional gain
        self.Kd = np.array(Kd)  # Derivative gain
        self.tau = None  # Initialize torque variable
    
    def update(self, q_r, q_r_dot, q_r_ddot):
        # Compute jointspace torque, return torque
        q = self.robot._q
        v = self.robot._v 
        model = self.robot._model
        data = model.createData()
        # PD control law
        M = pin.crba(model, data, q)  # Compute mass matrix
        b = pin.nonLinearEffects(model, data, q, v)
        self.tau = M @ (q_r_ddot + self.Kd * (q_r_dot - v) + self.Kp * (q_r - q)) + b
        return self.tau.flatten()  # Return the computed torque
    
class CartesianSpaceController:
    """CartesianSpaceController
    Tracking controller in cartspace
    """
    def __init__(self, robot, joint_name, Kp, Kd):
        # save gains, robot ref
        None
        
    def update(self, X_r, X_dot_r, X_ddot_r):
        # compute cartesian control torque, return torque
        None

################################################################################
# Application
################################################################################
    
class Environment(Node):
    def __init__(self):        
        # Initialize the ROS 2 node
        super().__init__('robot_environment')  # Add this line to initialize the Node properly
        
        # state
        self.cur_state = State.JOINT_SPLINE
        
        # create simulation
        self.simulator = PybulletWrapper()
        
        # joint state publisher
        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 10)
        self.get_logger().info("Joint state publisher created.")
        
        ########################################################################
        # spawn the robot
        ########################################################################
        self.q_home = np.zeros(32)
        self.q_home[14:22] = np.array([0, +0.45, 0, -1, 0, 0, 0, 0 ])
        self.q_home[22:30] = np.array([0, -0.45, 0, -1, 0, 0, 0, 0 ])
        

        self.q_init = np.zeros(32)

        # Initialize the Talos robot
        self.robot = Talos(
            self.simulator,
            q=self.q_init,
            verbose=True,
            useFixedBase=True
        )
        self.robot.joint_state_publisher = self.joint_state_publisher
        self.robot.node = self  # Set the node reference in the robot
        ########self.robot.update()  # Ensure the robot's state is updated

        ########################################################################
        # joint space spline: init -> home
        ########################################################################

        # TODO: create a joint spline 
        self.T_spline = 5.0  # Duration of the spline
        self.arm_joint_indices = list(range(14, 30)) # Indices for arm joints

        self.q_init_arms = self.q_init[self.arm_joint_indices]
        self.q_home_arms = self.q_home[self.arm_joint_indices]

        delta_q_arms = self.q_home_arms - self.q_init_arms
        
        self.A0_arms = self.q_init_arms
        self.A1_arms = np.zeros_like(delta_q_arms)
        self.A2_arms = 3 * delta_q_arms / (self.T_spline**2)
        self.A3_arms = -2 * delta_q_arms / (self.T_spline**3)

        # TODO: create a joint controller
          # Define 1D arrays for gains
        self.Kp_gains = np.zeros(self.robot.model.nq)
        self.Kd_gains = np.zeros(self.robot.model.nq)

        # Set gains for different robot segments
        for i in range(0, 12):  # Legs
            self.Kp_gains[i] = 930.0
            self.Kd_gains[i] = 1.0
        for i in range(12, 14):  # Torso
            self.Kp_gains[i] = 580.0
            self.Kd_gains[i] = 1.5
        for i in range(14, 30):  # Arms
            self.Kp_gains[i] = 280.0
            self.Kd_gains[i] = 0.5
        for i in range(30, 32):  # Head
            self.Kp_gains[i] = 50.0
            self.Kd_gains[i] = 0.5

        self.joint_space_controller = JointSpaceController(self.robot, self.Kp_gains, self.Kd_gains)

        ########################################################################
        # cart space: hand motion
        ########################################################################

        # TODO: create a cartesian controller
        
        ########################################################################
        # logging
        ########################################################################
        
        # TODO: publish robot state every 0.01 s to ROS
        self.t_publish = 0.0
        self.publish_period = 0.01
        
    def update(self, t, dt):
        
        # TODO: update the robot and model
        self.robot.update()
        #self.robot.publish()
        # update the controllers
        # TODO: Do inital jointspace, switch to cartesianspace control
        q_r = self.q_init.copy() # Default to initial pose for non-moving joints
        q_r_dot = np.zeros(self.robot.model.nq)
        q_r_ddot = np.zeros(self.robot.model.nq)

        # Check if we are in the joint spline state
        if t <= self.T_spline:
            t_current_spline = t
            q_r_arms = self.A0_arms + self.A1_arms * t_current_spline + self.A2_arms * (t_current_spline**2) + self.A3_arms * (t_current_spline**3)
            q_r_dot_arms = self.A1_arms + 2 * self.A2_arms * t_current_spline + 3 * self.A3_arms * (t_current_spline**2)
            q_r_ddot_arms = 2 * self.A2_arms + 6 * self.A3_arms * t_current_spline
        else: # After spline duration, hold arm home position
            q_r_arms = self.q_home_arms
            q_r_dot_arms = np.zeros_like(self.q_home_arms)
            q_r_ddot_arms = np.zeros_like(self.q_home_arms)
        # command the robot
        
        # Assign arm references to full reference vectors
        q_r[self.arm_joint_indices] = q_r_arms
        q_r_dot[self.arm_joint_indices] = q_r_dot_arms
        q_r_ddot[self.arm_joint_indices] = q_r_ddot_arms

        
      

        #tau = self.joint_space_controller.update(q_r, q_r_dot, q_r_ddot)
        #self.robot.setActuatedJointTorques(tau)
            
        # TODO: publish ros stuff
        self.robot.publish()
        

def main():    
    rclpy.init()  # Initialize ROS 2
    env = Environment()
    
    try:
        while rclpy.ok():  # ROS 2 equivalent of checking if the node is active
            t = env.simulator.simTime()
            dt = env.simulator.stepTime()
            
            env.update(t, dt)
            
            env.simulator.debug()
            env.simulator.step()
            rclpy.spin_once(env, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        env.destroy_node()  # Clean up the ROS 2 node
        rclpy.shutdown()  # Shutdown ROS 2

if __name__ == '__main__': 
    main()

