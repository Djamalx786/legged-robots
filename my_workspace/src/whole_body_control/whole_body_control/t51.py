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
from rclpy.executors import ExternalShutdownException
import tf2_ros
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped

################################################################################
# settings
################################################################################

DO_PLOT = True

################################################################################
# Robot
################################################################################


class Talos(Robot):
    def __init__(self, simulator, urdf, model, node, q=None, verbose=True, useFixedBase=True):
        # call base class constructor

        # Initial condition for the simulator an model
        z_init = 1.15

        super().__init__(
            simulator,
            urdf,
            model,
            basePosition=[0, 0, z_init],
            baseQuationerion=[0, 0, 0, 1],
            q=q,
            useFixedBase=False,
            verbose=verbose)

        self.node = node

        # add publisher
        self.pub_joint = self.node.create_publisher(
            JointState, "/joint_states", 10)

        self.joint_msg = JointState()
        self.joint_msg.name = self.actuatedJointNames()

        # add tf broadcaster
        self.br = tf2_ros.TransformBroadcaster(self.node)

        pb.enableJointForceTorqueSensor(self.id(), self.jointNameIndexMap()["leg_right_6_joint"], True)
        pb.enableJointForceTorqueSensor(self.id(), self.jointNameIndexMap()["leg_left_6_joint"], True)

    def update(self):
        # update base class
        super().update()

    def publish(self, T_b_w, tau):
        # publish jointstate
        self.joint_msg.header.stamp = self.node.get_clock().now().to_msg()
        self.joint_msg.position = self.actuatedJointPosition().tolist()
        self.joint_msg.velocity = self.actuatedJointVelocity().tolist()
        self.joint_msg.effort = tau.tolist()

        self.pub_joint.publish(self.joint_msg)

        # broadcast transformation T_b_w
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.node.get_clock().now().to_msg()
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = self.baseName()

        tf_msg.transform.translation.x = T_b_w.translation[0]
        tf_msg.transform.translation.y = T_b_w.translation[1]
        tf_msg.transform.translation.z = T_b_w.translation[2]

        q = pin.Quaternion(T_b_w.rotation)
        q.normalize()
        tf_msg.transform.rotation.x = q.x
        tf_msg.transform.rotation.y = q.y
        tf_msg.transform.rotation.z = q.z
        tf_msg.transform.rotation.w = q.w

        self.br.sendTransform(tf_msg)


################################################################################
# Application
################################################################################


class StateMachine:
    def __init__(self, robot, simulator):
        print("StateMachine initialized")
        self.robot = robot
        self.simulator = simulator
        self.state = "right"
        self.t_start = self.simulator.simTime()
        self.t_push = 1.0  # duration of force application
        self.t_period = 2.0  # interval between force applications
        self.force_magnitude = 15.0  # Increased force magnitude
        self.line_id = -1
        self.current_force = None  # Track current force being applied
        self.pushing = False  # Flag to track if currently pushing

    def update(self):
        t = self.simulator.simTime()
        elapsed = t - self.t_start

        if self.state == "right" and elapsed > self.t_period:
            self.start_pushing([self.force_magnitude, 0, 0])
            self.state = "right_pushing"
            self.t_start = t
        elif self.state == "right_pushing" and elapsed > self.t_push:
            self.stop_pushing()
            self.state = "left"
            self.t_start = t
        elif self.state == "left" and elapsed > self.t_period:
            self.start_pushing([-self.force_magnitude, 0, 0])
            self.state = "left_pushing"
            self.t_start = t
        elif self.state == "left_pushing" and elapsed > self.t_push:
            self.stop_pushing()
            self.state = "back"
            self.t_start = t
        elif self.state == "back" and elapsed > self.t_period:
            self.start_pushing([0, -self.force_magnitude, 0])
            self.state = "back_pushing"
            self.t_start = t
        elif self.state == "back_pushing" and elapsed > self.t_push:
            self.stop_pushing()
            self.state = "done"
            self.t_start = t

        # Apply force continuously while pushing
        if self.pushing and self.current_force is not None:
            self.apply_continuous_force()

    def start_pushing(self, force):
        """Start applying continuous force"""
        self.current_force = force
        self.pushing = True
        print(f"Starting continuous push with force: {force}")
        self.visualize_force(force, self.get_hip_position())

    def stop_pushing(self):
        """Stop applying force"""
        self.pushing = False
        self.current_force = None
        print("Stopping push")
        self.remove_force()

    def apply_continuous_force(self):
        """Apply force continuously every frame"""
        if self.current_force is not None:
            position = self.get_hip_position()
            self.robot.applyForce(self.current_force, position)

    def visualize_force(self, force, position):
        """Visualize the force direction using a debug line"""
        p1 = position
        p2 = p1 + np.array(force) / self.force_magnitude  # Scale for visualization
        self.line_id = self.simulator.addGlobalDebugLine(p1, p2, line_id=self.line_id, color=[1, 0, 0])

    def remove_force(self):
        """Remove the debug line"""
        if self.line_id != -1:
            self.simulator.removeDebugItem(self.line_id)
            self.line_id = -1

    def get_hip_position(self):
        """Get current hip/base position for visualization"""
        hip_position = self.robot.q()[:3]  
        return hip_position

class Environment(Node):
    def __init__(self):
        super().__init__('tutorial_4_standing_node')

        # init TSIDWrapper
        self.tsid_wrapper = TSIDWrapper(conf)

        # init Simulator
        self.simulator = PybulletWrapper()

        q_init = np.hstack([np.array([0, 0, 1.15, 0, 0, 0, 1]),
                           np.zeros_like(conf.q_actuated_home)])

        # init ROBOT
        self.robot = Talos(
            self.simulator,
            conf.urdf,
            self.tsid_wrapper.model,
            self,
            q=q_init,
            verbose=True,
            useFixedBase=False)

        self.t_publish = 0.0

        # init StateMachine
        self.state_machine = StateMachine(self.robot, self.simulator)

    def update(self):
        # elapsed time
        t = self.simulator.simTime()

        # update the simulator and the robot
        self.simulator.step()
        self.simulator.debug()
        self.robot.update()

        # Get robot state - ADD THIS LINE
        q = self.robot.q()

        # update TSID controller
        tau_sol, _ = self.tsid_wrapper.update(q, self.robot.v(), t)
        
        # Read Ankle Wrenches
        wren = pb.getJointState(self.robot.id(), self.robot.jointNameIndexMap()["leg_right_6_joint"])[2]
        wnp = np.array([-wren[0], -wren[1], -wren[2], -wren[3], -wren[4], -wren[5]])
        wr_rankle = pin.Force(wnp)
        wren = pb.getJointState(self.robot.id(), self.robot.jointNameIndexMap()["leg_left_6_joint"])[2]
        wnp = np.array([-wren[0], -wren[1], -wren[2], -wren[3], -wren[4], -wren[5]])
        wl_lankle = pin.Force(wnp)

        # Position of the ankles and the soles 
        data = self.robot._model.createData()
        pin.framesForwardKinematics(self.robot._model, data, q)  # Now q is defined
        H_w_lsole = data.oMf[self.robot._model.getFrameId("left_sole_link")]
        H_w_rsole = data.oMf[self.robot._model.getFrameId("right_sole_link")]
        H_w_lankle = data.oMf[self.robot._model.getFrameId("leg_left_6_joint")]
        H_w_rankle = data.oMf[self.robot._model.getFrameId("leg_right_6_joint")]

        print("Left Ankle Force:", wl_lankle.vector)  # Also add .vector to see the actual force values
        print("Left Sole Position:", H_w_lsole.translation)
        
        # command to the robot
        self.robot.setActuatedJointTorques(tau_sol)

        # update state machine
        self.state_machine.update()

        # publish to ros
        if t - self.t_publish > 1./30.:
            self.t_publish = t
            # get current BASE Pose
            T_b_w, _ = self.tsid_wrapper.baseState()
            self.robot.publish(T_b_w, tau_sol)


################################################################################
# main
################################################################################


def main(args=None):
    rclpy.init(args=args)
    env = Environment()
    try:
        while rclpy.ok():
            env.update()

    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        env.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
