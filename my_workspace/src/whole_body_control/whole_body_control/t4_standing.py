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
from geometry_msgs.msg import PoseStamped # Keep if used elsewhere, otherwise can be removed
from geometry_msgs.msg import TransformStamped # Added for TF broadcasting

################################################################################
# settings
################################################################################

DO_PLOT = True

################################################################################
# Robot
################################################################################

class Talos(Robot):
    def __init__(self, simulator, urdf, model, q=None, verbose=True, useFixedBase=True, node: Node = None): # Added node parameter
        # TODO call base class constructor
        super().__init__(simulator, urdf, model, q=q, verbose=verbose, useFixedBase=False) # useFixedBase=False as per instructions

        self.node = node
        if self.node:
            # TODO add publisher
            self.joint_state_publisher = self.node.create_publisher(JointState, 'joint_states', 10)
            # TODO add tf broadcaster
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self.node)
        else:
            if verbose:
                print("No ROS2 node provided to Talos robot. Skipping publisher/broadcaster setup.")
            self.joint_state_publisher = None
            self.tf_broadcaster = None


    def update(self):
        # TODO update base class
        super().update()
    
    def publish(self, T_b_w: pin.SE3):
        if not self.node:
            return

        current_time = self.node.get_clock().now().to_msg()

        # TODO publish jointstate
        if self.joint_state_publisher:
            js_msg = JointState()
            js_msg.header.stamp = current_time
            # Assuming self.actuated_joint_names and self.q are available from base Robot class
            # For a floating base robot, q[:7] is base pose, q[7:] are joint angles
            js_msg.name = self.actuatedJointNames()
            if self.q is not None and len(self.q) > 7:
                js_msg.position = self.q[7:].tolist()
            else:
                js_msg.position = [] # Or handle error appropriately
            # js_msg.velocity = ... # Optionally add velocity
            # js_msg.effort = ...   # Optionally add effort
            self.joint_state_publisher.publish(js_msg)
        
        # TODO broadcast transformation T_b_w
        if self.tf_broadcaster:
            tfs_msg = TransformStamped()
            tfs_msg.header.stamp = current_time
            tfs_msg.header.frame_id = "world"
            tfs_msg.child_frame_id = "base_link" 

            translation = T_b_w.translation
            rotation = pin.Quaternion(T_b_w.rotation) 

            tfs_msg.transform.translation.x = translation[0]
            tfs_msg.transform.translation.y = translation[1]
            tfs_msg.transform.translation.z = translation[2]

            tfs_msg.transform.rotation.x = rotation.x
            tfs_msg.transform.rotation.y = rotation.y
            tfs_msg.transform.rotation.z = rotation.z
            tfs_msg.transform.rotation.w = rotation.w
            
            self.tf_broadcaster.sendTransform(tfs_msg)

################################################################################
# main
################################################################################

def main():
    # Initialize ROS
    rclpy.init()

    # Create ROS node
    node = rclpy.create_node('t4_standing')

    # Initialize TSIDWrapper
    tsid_wrapper = TSIDWrapper(conf)

    # Initialize Simulator
    simulator = PybulletWrapper()

    # Initialize Robot
    robot = Talos(
        simulator=simulator,
        urdf=conf.urdf,
        model=tsid_wrapper.model,
        q=conf.q_home,
        useFixedBase=False,
        node=node
    )

    # Set initial posture reference
    tsid_wrapper.setPostureRef(conf.q_actuated_home)
    #robot.getlogger().info("TSID initialized with home posture.")

    # Main loop
    t_publish = 0.0
    while rclpy.ok():
        # Elapsed time
        t = simulator.simTime()

        # Step simulator and update robot state
        simulator.step()
        robot.update()
        q = robot._q 
        v = robot._v
        # Update TSID controller
        if q is not None and v is not None:  # Ensure robot state is valid
            sol = tsid_wrapper.update(q, v, t)
            tau_sol = tsid_wrapper.get_tau(sol)
            robot.setActuatedJointTorques(tau_sol)

        # Publish to ROS at 30 Hz
        if t - t_publish >= 1. / 30.:
            t_publish = t
            if sol is not None:  # Ensure TSID solution is valid
                T_b_w = tsid_wrapper.baseState(sol)
                robot.publish(T_b_w)

        # Spin ROS node
        rclpy.spin_once(node, timeout_sec=0.0001)

    # Shutdown ROS
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    rclpy.init()
    main()
    rclpy.shutdown()

