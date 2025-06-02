import numpy as np
import numpy.linalg as la

# simulator (#TODO: set your own import path!)
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot

# modeling
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from std_msgs.msg import String, Header

from enum import Enum

# ROS
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import numpy.linalg as la
from scipy.interpolate import CubicSpline

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
           
        
        self.urdf = "src/talos_description/robots/talos_reduced.urdf"
        self.path_meshes = "src/talos_description/meshes/../.."
        

        self.wrapper = pin.RobotWrapper.BuildFromURDF(self.urdf, self.path_meshes, None, True, None)

        self.model = self.wrapper.model
        self.data = self.wrapper.data

        print("RobotWrapper loaded with {} joints".format(self.model.nq))

        super().__init__(simulator,
                        self.urdf,
                        self.model,
                        [0, 0, 1.15],  # Floating base initial position
                        [0, 0, 0, 1],   # Floating base initial orientation [x,y,z,w]
                        q=q,
                        useFixedBase=useFixedBase,
                        verbose=verbose)
        
        # Publisher will be set from outside (placeholder for injection)
        self.joint_state_publisher = None
        self.node = None
        
        
    def update(self):
        
        super().update()
        
        self.wrapper.forwardKinematics(self._q)
        
    def massMatrix(self):
        """Compute and return the mass matrix for actuated joints only"""
        pin.forwardKinematics(self.model, self.data, self._q)
        
        # Compute mass matrix using Pinocchio
        M_full = pin.crba(self.model, self.data, self._q)
        
        # Return only actuated joints part
        if self._use_fixed_base:
            return M_full  # Should be 32x32
        else:
            return M_full[6:, 6:]  
    
    def coriolisAndGravity(self):
        """Compute and return Coriolis and gravity terms for actuated joints only"""
        pin.forwardKinematics(self.model, self.data, self._q, self._v)
        
        # Compute nonlinear effects (Coriolis + gravity)
        b = pin.nonLinearEffects(self.model, self.data, self._q, self._v)
        
        # Return only actuated joints part
        if self._use_fixed_base:
            return b  # Should be 32
        else:
            return b[6:]  
        
    def wrapper(self):
        return self._wrapper

    def data(self):
        return self._wrapper.data
    
    def publish(self):
        
        
        if self.joint_state_publisher is None or self.node is None:
            return
            
        

        current_actuated_joint_names = self.actuatedJointNames()
        num_act_joints = len(current_actuated_joint_names)

        ordered_q = np.zeros(num_act_joints)
        ordered_v = np.zeros(num_act_joints)

        for i in range(num_act_joints):
            name = current_actuated_joint_names[i]
            try:
                joint_id_pin = self.model.getJointId(name)
                idx_q_pin = self.model.joints[joint_id_pin].idx_q
                idx_v_pin = self.model.joints[joint_id_pin].idx_v
                
                ordered_q[i] = self._q[idx_q_pin]
                ordered_v[i] = self._v[idx_v_pin]
            except KeyError:
                if self.node:
                    self.node.get_logger().warn(f"Joint '{name}' from PyBullet actuated list not found in Pinocchio model during publishing, or issue with indexing.")
                ordered_q[i] = 0.0 # Default to 0 if error
                ordered_v[i] = 0.0
        
        tau = np.zeros(num_act_joints)  # Efforts should also match this order
        
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = self.node.get_clock().now().to_msg()
        joint_state_msg.name = current_actuated_joint_names
        joint_state_msg.position = ordered_q.tolist()
        joint_state_msg.velocity = ordered_v.tolist()
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
        self.Kp = np.diag(Kp)
        self.Kd = np.diag(Kd)

    def update(self, q_r, q_r_dot, q_r_ddot):
        # Compute jointspace torque, return torque
        #τ = M (q) [q̈r − Kd (q̇ − q̇r ) − Kp (q − qr )] + h (q, q̇)
        
        if self.robot._use_fixed_base:
            q = self.robot._q
            v = self.robot._v
        else:
            q = self.robot._q[self.robot._pos_idx_offset:]  # Skip floating base position (7 DOF)
            v = self.robot._v[self.robot._vel_idx_offset:]  # Skip floating base velocity (6 DOF)
            
    

        position_error = q_r - q
        velocity_error = q_r_dot - v

        # Compute the mass matrix
        M = self.robot.massMatrix()
        h = self.robot.coriolisAndGravity()
        

        # Compute the desired acceleration ( sign change for consistancy/best practice)
        tau = M @ (q_r_ddot + self.Kd @ velocity_error + self.Kp @ position_error) + h

        return tau

class CartesianSpaceController:
    """CartesianSpaceController
    Tracking controller in cartspace
    """
    def __init__(self, robot, joint_name, Kp, Kd):
        # save gains, robot ref
        self.robot = robot
        self.joint_name = joint_name
        if len(Kp) > 6:
            # Take only the first 6 elements for Cartesian control
            self.Kp = np.diag(Kp[:6])
            self.Kd = np.diag(Kd[:6])
        else:
            self.Kp = np.diag(Kp)
            self.Kd = np.diag(Kd)# Only use first 6 DOF for Cartesian control

        self.id = robot.model.getJointId(self.joint_name)
        
        
        if robot._use_fixed_base:
            self.q_posture_ref = np.zeros(32)
            self.q_posture_ref[14:22] = np.array([0, +0.45, 0, -1, 0, 0, 0, 0])  # left arm
            self.q_posture_ref[22:30] = np.array([0, -0.45, 0, -1, 0, 0, 0, 0])  # right arm
        else:
            self.q_posture_ref = np.zeros(robot.actuatedJointCount())
            

    def update(self, X_r, X_dot_r, X_ddot_r):

        if self.robot._use_fixed_base:
            q = self.robot._q
            v = self.robot._v
        else:
            # For floating base robot, use the correct offsets
            #τ = MJ# Ẍd − J̇q + h
            # ̈X d = Ẍ r − Kd (Ẋ − Ẋ )− Kp (X − X r)
            q = self.robot._q[self.robot._pos_idx_offset:]  # Skip floating base position (7 DOF)
            v = self.robot._v[self.robot._vel_idx_offset:] 

        J = pin.computeJointJacobian(self.robot.model, self.robot.data, self.robot._q, self.id)
        
        if not self.robot._use_fixed_base:
            J = J[:, 6:]
        
        X = self.robot.data.oMi[self.id]
        X_dot = J @ v
        X_error = pin.log6(X.inverse() * X_r).vector 
        X_dot_error = X_dot_r - X_dot
        X_ddot_d = X_ddot_r + self.Kd @ X_dot_error + self.Kp @ X_error
        
        J_dot_q_dot = pin.getClassicalAcceleration(self.robot.model, self.robot.data, self.id, pin.ReferenceFrame.LOCAL).vector
        
        if not self.robot._use_fixed_base:
            # Get only the actuated part of the acceleration
            a_actuated = np.zeros(len(v))  # Initialize with zeros for actuated joints
            J_dot_q_dot = J @ a_actuated  # This will be zero for this implementation
            
        damp = 1e-6
        J_pinv = J.T @ np.linalg.inv(J @ J.T + damp * np.eye(6))
        
        n_joints = len(q)
        N = np.eye(n_joints) - J.T @ J_pinv.T
        
        q_ddot_cart = J_pinv @ (X_ddot_d - J_dot_q_dot)
        
        position_error = self.q_posture_ref - q
        velocity_error = np.zeros(n_joints) - v 
        
        
        
        Kp_posture = 60.0  
        Kd_posture = 0.5
        q_ddot_posture = Kp_posture * position_error + Kd_posture * velocity_error
        
        M = self.robot.massMatrix()
        h = self.robot.coriolisAndGravity()

        

        tau = M @ (q_ddot_cart + N @ q_ddot_posture) + h
        
        

        return tau

################################################################################
# Application
################################################################################
    
class Environment(Node):
    def __init__(self):
        super().__init__('environment_talos')
                        
        # state
        self.cur_state = State.JOINT_SPLINE
        
        # create simulation
        self.simulator = PybulletWrapper()
        
        self.joint_state_publisher = self.create_publisher(
            JointState, 
            'joint_states', 
            10)
        
        self.pose_subscriber= self.create_subscription(
            PoseStamped,
            'hand_target_pose',
            self.traget_pose_callback,
            10)
        
        ########################################################################
        # spawn the robot
        ########################################################################
        self.q_home = np.zeros(32)
        self.q_home[14:22] = np.array([0, +0.45, 0, -1, 0, 0, 0, 0 ])
        self.q_home[22:30] = np.array([0, -0.45, 0, -1, 0, 0, 0, 0 ])
    
        
        self.q_init = np.zeros(32)
        
        self.robot = Talos(self.simulator, q=self.q_init, verbose=True, useFixedBase=True)

        self.robot.joint_state_publisher = self.joint_state_publisher
        self.robot.node = self

        ########################################################################
        # joint space spline: init -> home
        ########################################################################

        # TODO: create a joint spline 
        # TODO: create a joint controller
        self.spline_duration = 5.0
        t_points = np.array([0.0, self.spline_duration])
        
        if self.robot._use_fixed_base:
            q_ini_ac = self.robot._q
            q_home_ac = self.q_home
            n_actuated = len(q_ini_ac)
        else:
            q_ini_ac = self.robot._q[self.robot._pos_idx_offset:]  # Use robot's position offset
            q_home_ac = self.q_home[self.robot._pos_idx_offset:]   # Use robot's position offset
            n_actuated = len(q_ini_ac)

        q_waypoints = np.column_stack([q_ini_ac, q_home_ac])
        
        self.joint_spline = CubicSpline(t_points, q_waypoints.T, bc_type='clamped')
        
        kp_base = 70.0
        kd_base = 3.0
        
        # Initialize gain arrays for actuated joints (32 joints)
        Kp = np.zeros(32)
        Kd = np.zeros(32)
        
        # Set gains for different robot segments
        for i in range(0, 12):  # Legs
            Kp[i] = 930.0
            Kd[i] = 1.0
        for i in range(12, 14):  # Torso
            Kp[i] = 580.0
            Kd[i] = 1.5
        for i in range(14, 30):  # Arms
            Kp[i] = 280.0
            Kd[i] = 0.5
        for i in range(30, 32):  # Head
            Kp[i] = 50.0
            Kd[i] = 0.5

        self.joint_controller = JointSpaceController(self.robot, Kp, Kd)
        
        ########################################################################
        # cart space: hand motion
        ########################################################################

        # TODO: create a cartesian controller
        self.cartesian_controller = CartesianSpaceController(self.robot, "arm_right_7_joint", Kp, Kd)
        
        self.X_goal = None


        ########################################################################
        # logging
        ########################################################################
        
        self.t_publish = 0.0
        self.publish_period = 0.01
        
    def traget_pose_callback(self, msg):
        
        pos = msg.pose.position
        orientation = msg.pose.orientation
        position = np.array([pos.x, pos.y, pos.z])
        orientation = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
        self.X_goal = pin.XYZQUATToSE3(np.concatenate([position, orientation]))
        self.get_logger().debug(f"Received target pose: {self.X_goal}")
        
    def update(self, t, dt):
        
        # TODO: update the robot and model
        self.robot.update()

        # update the controllers
        # TODO: Do inital jointspace, switch to cartesianspace control
        
        # command the robot
        
        if self.cur_state == State.JOINT_SPLINE:
            if t <= self.spline_duration:
                q_r = self.joint_spline(t)
                q_r_dot = self.joint_spline(t, 1)
                q_r_ddot = self.joint_spline(t, 2)
                tau = self.joint_controller.update(q_r, q_r_dot, q_r_ddot)
            else:
                if self.X_goal is None:
                    hand_joint_id = self.robot.model.getJointId("arm_right_7_joint")
                    self.X_goal = self.robot.data.oMi[hand_joint_id].copy()
                    self.cartesian_controller.q_posture_ref = self.robot._q.copy()
                    print("Switching to Cartesian control, saved goal pose")
                    
                self.cur_state = State.CART_SPLINE
                
                q_r = self.joint_spline(self.spline_duration)
                q_r_dot = np.zeros_like(q_r)
                q_r_ddot = np.zeros_like(q_r)
                tau = self.joint_controller.update(q_r, q_r_dot, q_r_ddot)
                
                
        elif self.cur_state == State.CART_SPLINE:
            # Cartesian space control - hold the saved pose
            X_r = self.X_goal
            X_dot_r = np.zeros(6)
            X_ddot_r = np.zeros(6)
            
            tau = self.cartesian_controller.update(X_r, X_dot_r, X_ddot_r)

                
        else:
            tau = np.zeros(self.robot.actuatedJointCount())
        
        self.robot.setActuatedJointTorques(tau)

        if t - self.t_publish >= self.publish_period:
            self.robot.publish()
            self.t_publish = t

        
def main():
    rclpy.init()  
    env = Environment()
    
    try:
        while rclpy.ok():
            t = env.simulator.simTime()
            dt = env.simulator.stepTime()
            
            env.update(t, dt)
            
            env.simulator.debug()
            env.simulator.step()
            
            # Spin ROS callbacks
            rclpy.spin_once(env, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        env.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__': 
    main()
    