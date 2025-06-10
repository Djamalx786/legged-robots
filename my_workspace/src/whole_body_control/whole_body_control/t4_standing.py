import numpy as np
from numpy import nan
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt

# pinocchio
import pinocchio as pin

# simulator
import pybullet as pb
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot


# robot and controller
from whole_body_control.tsid_wrapper import TSIDWrapper
import whole_body_control.config as conf

# ROS
import rclpy
from rclpy.node import Node
import tf2_ros
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Header
from geometry_msgs.msg import PoseStamped # Keep if used elsewhere, otherwise can be removed
from geometry_msgs.msg import TransformStamped # Added for TF broadcasting

################################################################################
# settings
################################################################################

DO_PLOT = True

################################################################################
# Robot
################################################################################

class Talos(Robot, Node):
    def __init__(self, simulator, urdf, model, q=None, verbose=True, useFixedBase=False, node: Node = None):
        Node.__init__(self, 'talos_controller_node')

        initial_base_pos = np.array([0.0, 0.0, 1.1])  # Default height
        initial_base_quat = np.array([0.0, 0.0, 0.0, 1.0])

        #if q is not None and len(q) >= 7:
        #    initial_base_pos = q[0:3]
        #    initial_base_quat = q[3:7]

        print(f"Initial base position: {initial_base_pos}")  # Debugging-Ausgabe

        Robot.__init__(self,
                       simulator,
                       filename=urdf,
                       model=model,
                       basePosition=initial_base_pos,
                       baseQuationerion=initial_base_quat,
                       q=q,
                       useFixedBase=False,
                       verbose=verbose)
        
        # ROS Publishers and Broadcasters
        self.joint_state_publisher = self.create_publisher(
            JointState, 
            'joint_states', 
            10)
        
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

    def update(self):
        # TODO update base class
        super().update()
    
    def publish(self):
        # Retrieve the current robot configuration and velocity
        current_q_pin = self.q() 
        
        now = self.get_clock().now().to_msg()
        
        # Prepare the JointState message for publishing
        js_msg = JointState()
        js_msg.header = Header()
        js_msg.header.stamp = now
        
        # Extract joint names, positions, and velocities
        joint_names = self.actuatedJointNames()
        joint_positions = self.actuatedJointPosition()
        joint_velocities = self.actuatedJointVelocity()
        
        # Handle torque commands, defaulting to zeros if unavailable
        if hasattr(self, '_tau_cmd') and self._tau_cmd is not None:
            joint_efforts = self._tau_cmd
        else:
            joint_efforts = np.zeros(len(joint_names))
        
        js_msg.name = joint_names
        js_msg.position = joint_positions.tolist()
        js_msg.velocity = joint_velocities.tolist()
        js_msg.effort = joint_efforts.tolist()
            
        self.joint_state_publisher.publish(js_msg)
        
        # Send a transform for the robot's base link relative to the world frame
        base_translation = current_q_pin[0:3]
        base_orientation_quat = current_q_pin[3:7]  # Quaternion format: [qx, qy, qz, qw]

        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "world"  # Reference frame
        t.child_frame_id = self.baseName()  # Robot's base link name

        t.transform.translation.x = float(base_translation[0])
        t.transform.translation.y = float(base_translation[1])
        t.transform.translation.z = float(base_translation[2])

        t.transform.rotation.x = float(base_orientation_quat[0])
        t.transform.rotation.y = float(base_orientation_quat[1])
        t.transform.rotation.z = float(base_orientation_quat[2])
        t.transform.rotation.w = float(base_orientation_quat[3])
        
        self.tf_broadcaster.sendTransform(t)
################################################################################
# main
################################################################################

def main():
    print("Starting main function...")
    rclpy.init()
    print("ROS initialized...")

    # Initialize TSIDWrapper
    print("Initializing TSIDWrapper...")
    tsid_wrapper = TSIDWrapper(conf)
    print("TSIDWrapper initialized...")

    # Initialize Simulator
    print("Initializing Simulator...")
    simulator = PybulletWrapper(sim_rate=1000)
    print("Simulator initialized...")

    # Initialize Robot
    print("Initializing Robot...")
    robot = Talos(
        simulator=simulator,
        urdf=conf.urdf,
        model=tsid_wrapper.model,
        q=conf.q_home,
    )
    print("Robot initialized...")

    # Set initial posture reference
    tsid_wrapper.setPostureRef(conf.q_actuated_home)
    #robot.getlogger().info("TSID initialized with home posture.")
    tau = np.zeros(robot.numActuatedJoints())
    # Main loop
    t_publish = 0.0
    try: 
        while rclpy.ok():
            rclpy.spin_once(robot, timeout_sec=0.0001)
            # Elapsed time
            t = simulator.simTime()

            # Step simulator and update robot state
            simulator.step()
            robot.update()
            q = robot.q()
            v = robot.v()
            robot.setActuatedJointTorques(tau)

            # Update TSID controller
            
            if q is not None and v is not None:  # Ensure robot state is valid
                tau_sol, dv_sol = tsid_wrapper.update(q, v, t)
                robot.setActuatedJointTorques(tau_sol)

            # Publish to ROS at 30 Hz
            if t - t_publish >= 1. / 30.:
                t_publish = t
                if dv_sol is not None:  # Ensure TSID solution is valid
                    T_b_w = tsid_wrapper.baseState(dv_sol)
                    robot.publish()
            
            
            
    
    except KeyboardInterrupt:
        robot.get_logger().info("Keyboard interrupt received, shutting down.")
    
    finally:
        # Cleanly shutdown ROS
        robot.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

