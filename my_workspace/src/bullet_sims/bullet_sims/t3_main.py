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
from scipy.interpolate import CubicSpline

################################################################################
# utility functions
################################################################################

class State(Enum):
    JOINT_SPLINE = 0,
    CART_SPLINE = 1
################################################################################

################################################################################

class Talos(Robot):
    def __init__(self, simulator, q=None, verbose=True, useFixedBase=True):
      
        self.urdf = "src/talos_description/robots/talos_reduced.urdf"
        self.path_meshes = "src/talos_description/meshes/../.."
        # Initialize Pinocchio robot wrapper
        self.wrapper = pin.RobotWrapper.BuildFromURDF(self.urdf, self.path_meshes, None, True, None)

        self.model = self.wrapper.model
        self.data = self.wrapper.data

        if verbose:
            print(f"Pinocchio RobotWrapper initialized with {self.model.nq} degrees of freedom")

        super().__init__(simulator,
                        self.urdf,
                        self.model,
                        [0, 0, 1.15],  # Base initial position
                        [0, 0, 0, 1],   # Base initial orientation [x,y,z,w]
                        q=q,
                        useFixedBase=True,
                        verbose=verbose)
        
        # ROS publishing components (to be injected externally)
        self.joint_state_publisher = None
        self.node = None
        
    def update(self):
        """Update robot state and forward kinematics"""
        super().update()
        
        # Update Pinocchio forward kinematics with current configuration
        self.wrapper.forwardKinematics(self._q)
        pin.updateFramePlacements(self.model, self.data)
        
    def massMatrix(self):
        """Compute mass matrix for actuated joints using CRBA algorithm"""
        # Ensure forward kinematics are computed
        pin.forwardKinematics(self.model, self.data, self._q)
        
        # Calculate full mass matrix using Composite Rigid Body Algorithm
        full_mass_matrix = pin.crba(self.model, self.data, self._q)
        
        # Extract actuated joints portion
        if self._use_fixed_base:
            return full_mass_matrix  # Fixed base: all joints are actuated
        else:
            return full_mass_matrix[6:, 6:]  # Floating base: skip first 6 DOF
    
    def coriolisAndGravity(self):
        """Compute Coriolis, centrifugal and gravitational terms for actuated joints"""
        # Update kinematics with positions and velocities
        pin.forwardKinematics(self.model, self.data, self._q, self._v)
        
        # Explicitly compute gravity and Coriolis terms for clarity
        pin.computeGeneralizedGravity(self.model, self.data, self._q)
        pin.computeCoriolisMatrix(self.model, self.data, self._q, self._v)
        
        # Get combined nonlinear effects vector
        nonlinear_terms = pin.nonLinearEffects(self.model, self.data, self._q, self._v)
        
        # Return actuated joints portion only
        if self._use_fixed_base:
            return nonlinear_terms  # Fixed base: return full vector
        else:
            return nonlinear_terms[6:]  # Floating base: skip base DOF
        
    def wrapper(self):
        """Return Pinocchio wrapper instance"""
        return self.wrapper

    def data(self):
        """Return Pinocchio data instance"""
        return self.wrapper.data
    
    def publish(self):
        """Publish current joint states to ROS topic"""
        
        if self.joint_state_publisher is None or self.node is None:
            return
            
        # Get actuated joint information
        actuated_joint_names = self.actuatedJointNames()
        num_actuated_joints = len(actuated_joint_names)

        # Initialize ordered arrays for joint states
        q_ordered_actuated = np.zeros(num_actuated_joints)
        v_ordered_actuated = np.zeros(num_actuated_joints)

        # Map joint states from Pinocchio ordering to PyBullet ordering
        for i in range(num_actuated_joints):
            joint_name = actuated_joint_names[i]
            try:
                # Get Pinocchio joint information
                joint_id_pinocchio = self.model.getJointId(joint_name)
                q_index_pinocchio = self.model.joints[joint_id_pinocchio].idx_q
                v_index_pinocchio = self.model.joints[joint_id_pinocchio].idx_v
                
                # Extract corresponding values
                q_ordered_actuated[i] = self._q[q_index_pinocchio]
                v_ordered_actuated[i] = self._v[v_index_pinocchio]
            except KeyError:
                if self.node:
                    self.node.get_logger().warn(f"Joint '{joint_name}' mapping error during state publishing")
                q_ordered_actuated[i] = 0.0 
                v_ordered_actuated[i] = 0.0
        
        # Initialize effort array (currently not measured)
        effort_ordered_actuated = np.zeros(num_actuated_joints)
        
        # Create and populate ROS message
        js_msg = JointState()
        js_msg.header = Header()
        js_msg.header.stamp = self.node.get_clock().now().to_msg()
        js_msg.name = actuated_joint_names
        js_msg.position = q_ordered_actuated.tolist()
        js_msg.velocity = v_ordered_actuated.tolist()
        js_msg.effort = effort_ordered_actuated.tolist()
        
        self.joint_state_publisher.publish(js_msg)

################################################################################
# Control Systems
################################################################################

class JointSpaceController:
    """
    Joint space tracking controller using computed torque method
    Control law: τ = M(q)[q̈_ref + Kd*ė + Kp*e] + h(q,q̇)
    """
    def __init__(self, robot_model, Kp , Kd):        
        # Store robot reference and control gains
        self.robot_reference = robot_model
        self.kp = np.diag(Kp)
        self.kd = np.diag(Kd)

    def update(self, q_r, q_r_dot, q_r_ddot):
        """Compute joint torques for trajectory tracking"""
        
        # Extract current actuated joint states
        if self.robot_reference._use_fixed_base:
            current_positions = self.robot_reference._q
            current_velocities = self.robot_reference._v
        else:
            current_positions = self.robot_reference._q[self.robot_reference._pos_idx_offset:]
            current_velocities = self.robot_reference._v[self.robot_reference._vel_idx_offset:]
            
        # Calculate tracking errors
        position_tracking_error = q_r - current_positions
        velocity_tracking_error = q_r_dot - current_velocities

        # Obtain robot dynamics components
        inertia_matrix = self.robot_reference.massMatrix()
        nonlinear_dynamics = self.robot_reference.coriolisAndGravity()
        
        # Compute desired acceleration with PD compensation
        acceleration_command = q_r_ddot + self.kd @ velocity_tracking_error + self.kp @ position_tracking_error
        
        # Apply computed torque control law
        commanded_torques = inertia_matrix @ acceleration_command + nonlinear_dynamics

        return commanded_torques

class CartesianSpaceController:
    """
    Cartesian space controller with null-space posture regulation
    Uses operational space control with dynamically consistent pseudo-inverse
    """
    def __init__(self, robot_model, target_link_name, Kp, Kd):
        # Store robot and end-effector information
        self.robot_reference = robot_model
        self.target_link_name = target_link_name
        
        # Configure Cartesian space gains (6 DOF: position + orientation)
        if len(Kp) > 6:
            # Extract first 6 elements if full joint gains provided
            self.cartesian_kp = np.diag(Kp[:6])
            self.cartesian_kd = np.diag(Kd[:6])
        else:
            self.cartesian_kp = np.diag(Kp)
            self.cartesian_kd = np.diag(Kd)

        # Get target link ID in Pinocchio model
        self.target_link_id = robot_model.model.getJointId(self.target_link_name)
        
        # Initialize default posture for null-space control
        if robot_model._use_fixed_base:
            self.posture_reference = np.zeros(32)
            # Set default arm configurations
            self.posture_reference[14:22] = np.array([0, +0.45, 0, -1, 0, 0, 0, 0])  # left arm
            self.posture_reference[22:30] = np.array([0, -0.45, 0, -1, 0, 0, 0, 0])  # right arm
        else:
            self.posture_reference = np.zeros(len(robot_model.actuatedJointNames()))

    def update(self, target_pose_se3, target_velocity_6d, target_acceleration_6d):
        """Compute torques for Cartesian space tracking with posture control"""

        # Get current actuated joint configuration and velocities
        if self.robot_reference._use_fixed_base:
            current_joint_positions = self.robot_reference._q
            current_joint_velocities = self.robot_reference._v
        else:
            current_joint_positions = self.robot_reference._q[self.robot_reference._pos_idx_offset:]
            current_joint_velocities = self.robot_reference._v[self.robot_reference._vel_idx_offset:]

        # Compute task Jacobian matrix
        full_jacobian = pin.computeJointJacobian(self.robot_reference.model, self.robot_reference.data, 
                                               self.robot_reference._q, self.target_link_id)
        
        if self.robot_reference._use_fixed_base:
            task_jacobian = full_jacobian
        else:
            # Extract actuated joint columns for floating base
            task_jacobian = full_jacobian[:, 6:]
        
        # Get current end-effector state
        current_pose_se3 = self.robot_reference.data.oMi[self.target_link_id]
        current_velocity_6d = task_jacobian @ current_joint_velocities
        
        # Calculate task space errors
        pose_error_6d = pin.log6(current_pose_se3.inverse() * target_pose_se3).vector
        velocity_error_6d = target_velocity_6d - current_velocity_6d
        
        # Compute desired task acceleration using PD law
        desired_task_acceleration = target_acceleration_6d + self.cartesian_kd @ velocity_error_6d + self.cartesian_kp @ pose_error_6d
        
        # Calculate Jacobian derivative term using Pinocchio
        jacobian_dot_q_dot = pin.getClassicalAcceleration(self.robot_reference.model, self.robot_reference.data, 
                                                        self.target_link_id, pin.ReferenceFrame.LOCAL).vector
        
        # Compute dynamically consistent pseudo-inverse
        regularization_term = 1e-6
        jacobian_pinv = task_jacobian.T @ np.linalg.inv(task_jacobian @ task_jacobian.T + regularization_term * np.eye(6))
        
        # Calculate task space joint accelerations
        task_accelerations = jacobian_pinv @ (desired_task_acceleration - jacobian_dot_q_dot)
        
        # Null-space projection for posture control
        num_actuated_joints = len(current_joint_positions)
        null_space_projector = np.eye(num_actuated_joints) - jacobian_pinv @ task_jacobian
        
        # Posture error computation (desired posture velocity = 0)
        posture_position_error = self.posture_reference - current_joint_positions
        posture_velocity_error = -current_joint_velocities  # Desired velocity is zero
        
        # Null-space PD control gains
        posture_kp_gain = 56.0  
        posture_kd_gain = 1.0
        
        # Compute posture control accelerations
        posture_accelerations = posture_kp_gain * posture_position_error + posture_kd_gain * posture_velocity_error
        
        # Get robot dynamics
        mass_matrix = self.robot_reference.massMatrix()
        coriolis_gravity = self.robot_reference.coriolisAndGravity()

        # Combine task and null-space accelerations
        total_accelerations = task_accelerations + null_space_projector @ posture_accelerations
        
        # Apply inverse dynamics
        control_torques = mass_matrix @ total_accelerations + coriolis_gravity

        return control_torques

################################################################################
# Main Application
################################################################################
    
class Environment(Node):
    def __init__(self):
        super().__init__('talos_control_environment')
                        
        # Control state machine
        self.current_control_mode = State.JOINT_SPLINE
        
        # Initialize simulation environment
        self.simulator = PybulletWrapper()
        
        # ROS publishers and subscribers
        self.joint_state_publisher = self.create_publisher(
            JointState, 
            'joint_states', 
            10)
        
        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            'hand_target_pose',
            self.target_pose_callback,  # Fixed typo
            10)
        
        ########################################################################
        # Robot initialization
        ########################################################################
        # Define target and initial configurations
        self.home_configuration = np.zeros(32)
        self.home_configuration[14:22] = np.array([0, +0.45, 0, -1, 0, 0, 0, 0])
        self.home_configuration[22:30] = np.array([0, -0.45, 0, -1, 0, 0, 0, 0])
        
        self.initial_configuration = np.zeros(32)
        
        # Create robot instance
        self.robot = Talos(self.simulator, q=self.initial_configuration, verbose=True, useFixedBase=True)

        # Inject ROS components into robot
        self.robot.joint_state_publisher = self.joint_state_publisher
        self.robot.node = self

        ########################################################################
        # Joint space trajectory planning
        ########################################################################
        # Configure trajectory duration and waypoints
        self.trajectory_duration = 5.0
        trajectory_time_points = np.array([0.0, self.trajectory_duration])
        
        # Set up trajectory waypoints based on robot configuration
        if self.robot._use_fixed_base:
            initial_actuated_config = self.robot._q
            target_actuated_config = self.home_configuration
        else:
            initial_actuated_config = self.robot._q[self.robot._pos_idx_offset:]
            target_actuated_config = self.home_configuration[self.robot._pos_idx_offset:]

        # Create trajectory waypoint matrix
        trajectory_waypoints = np.column_stack([initial_actuated_config, target_actuated_config])
        
        # Generate cubic spline trajectory
        self.joint_trajectory_spline = CubicSpline(trajectory_time_points, trajectory_waypoints.T, bc_type='clamped')
        
        # Configure joint space control gains
        Kp = np.zeros(32)
        Kd = np.zeros(32)
        
        # Set segment-specific gains
        for i in range(0, 12):  # Leg joints
            Kp[i] = 930.0
            Kd[i] = 1.0  # Fixed: was Kp[i] = 1.0
        for i in range(12, 14):  # Torso joints
            Kp[i] = 580.0
            Kd[i] = 1.5
        for i in range(14, 30):  # Arm joints
            Kp[i] = 280.0
            Kd[i] = 0.5
        for i in range(30, 32):  # Head joints
            Kp[i] = 50.0
            Kd[i] = 0.5

        # Initialize joint space controller
        self.joint_space_controller = JointSpaceController(self.robot, Kp, Kd)
        
        ########################################################################
        # Cartesian space control setup
        ########################################################################
        # Initialize Cartesian controller for right hand
        self.cartesian_space_controller = CartesianSpaceController(self.robot, "arm_right_7_joint", 
                                                                 Kp, Kd)
        
        # Target pose placeholder
        self.target_end_effector_pose = None

        ########################################################################
        # Publishing and timing configuration
        ########################################################################
        self.last_publish_time = 0.0
        self.publish_time_interval = 0.01  # 100 Hz
        
    def target_pose_callback(self, pose_message):
        """Handle incoming target pose messages"""
        
        # Extract position and orientation from ROS message
        position_ros = pose_message.pose.position
        orientation_ros = pose_message.pose.orientation
        
        # Convert to numpy arrays
        position_array = np.array([position_ros.x, position_ros.y, position_ros.z])
        orientation_array = np.array([orientation_ros.x, orientation_ros.y, orientation_ros.z, orientation_ros.w])
        
        # Convert to Pinocchio SE3 representation
        self.target_end_effector_pose = pin.XYZQUATToSE3(np.concatenate([position_array, orientation_array]))
        self.get_logger().debug(f"Received target end-effector pose: {self.target_end_effector_pose}")
        
    def update(self, simulation_time, time_step):
        """Main control loop update function"""
        
        # Update robot state and kinematics
        self.robot.update()

        # Initialize control torques
        control_torques = np.zeros(len(self.robot.actuatedJointNames()))  # Fixed: use len()
        
        # State machine for control modes
        if self.current_control_mode == State.JOINT_SPLINE:
            if simulation_time <= self.trajectory_duration:
                # Execute joint space trajectory tracking
                desired_positions = self.joint_trajectory_spline(simulation_time)
                desired_velocities = self.joint_trajectory_spline(simulation_time, 1)
                desired_accelerations = self.joint_trajectory_spline(simulation_time, 2)
                control_torques = self.joint_space_controller.update(desired_positions, desired_velocities, desired_accelerations)
            else:
                # Prepare for Cartesian control mode transition
                if self.target_end_effector_pose is None:
                    # Save current end-effector pose as target
                    hand_joint_id = self.robot.model.getJointId("arm_right_7_joint")
                    self.target_end_effector_pose = self.robot.data.oMi[hand_joint_id].copy()
                    
                    # Update posture reference for Cartesian controller
                    self.cartesian_space_controller.posture_reference = self.robot._q.copy()
                    self.get_logger().info("Transitioning to Cartesian control mode")
                    
                self.current_control_mode = State.CART_SPLINE  # Fixed: use correct enum value
                
                # Hold final trajectory position
                final_positions = self.joint_trajectory_spline(self.trajectory_duration)
                final_velocities = np.zeros_like(final_positions)
                final_accelerations = np.zeros_like(final_positions)
                control_torques = self.joint_space_controller.update(final_positions, final_velocities, final_accelerations)
                
        elif self.current_control_mode == State.CART_SPLINE:  # Fixed: use correct enum value
            # Execute Cartesian space control
            if self.target_end_effector_pose is not None:
                target_pose = self.target_end_effector_pose
                target_velocity = np.zeros(6)  # Hold position: zero velocity
                target_acceleration = np.zeros(6)  # Hold position: zero acceleration
                
                control_torques = self.cartesian_space_controller.update(target_pose, target_velocity, target_acceleration)
            else:
                self.get_logger().warn("Cartesian mode active but no target pose available")
                control_torques = np.zeros(len(self.robot.actuatedJointNames()))  # Fixed: use len()
        else:
            # Default case: no control
            control_torques = np.zeros(len(self.robot.actuatedJointNames()))  # Fixed: use len()
        
        # Apply computed torques to robot
        self.robot.setActuatedJointTorques(control_torques)

        # Publish joint states at specified frequency
        if simulation_time - self.last_publish_time >= self.publish_time_interval:
            self.robot.publish()
            self.last_publish_time = simulation_time

        
def main():
    rclpy.init()  
    environment_controller = Environment()
    
    try:
        while rclpy.ok():
            current_time = environment_controller.simulator.simTime()
            delta_time = environment_controller.simulator.stepTime()
            
            environment_controller.update(current_time, delta_time)
            
            environment_controller.simulator.debug()
            environment_controller.simulator.step()
            
            # Process ROS callbacks
            rclpy.spin_once(environment_controller, timeout_sec=0.0)
    except KeyboardInterrupt:
        environment_controller.get_logger().info("Shutdown requested by user")
    finally:
        environment_controller.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__': 
    main()
