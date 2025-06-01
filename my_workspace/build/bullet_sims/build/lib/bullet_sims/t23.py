import pybullet as pb
import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot

# TalosSimulator class for simulating the Talos robot
class TalosSimulator(Node):
    def __init__(self):
        super().__init__('talos_simulator')  # Initialize ROS2 node

        # Publisher for joint states
        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Simulation parameters
        self.simulation_timestep = 1.0 / 1000.0  # Simulation timestep
        self.trajectory_duration = 2.0  # Duration for trajectory
        self.publish_counter = 0
        self.publish_interval = int((1.0 / 30.0) / self.simulation_timestep)  # Steps for 30 Hz publishing

        # Paths for URDF and meshes
        self.urdf_path = "src/talos_description/robots/talos_reduced.urdf"
        self.mesh_path = "src/talos_description/meshes/../.."
                
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

        # Initialize robot configuration
        self._initialize_robot_configuration()

        # Initialize simulator and robot
        self._initialize_simulator()

        # Setup controller and trajectory
        self._setup_controller()
        self._setup_trajectory()

        # Timer for simulation steps
        self.timer = self.create_timer(self.simulation_timestep, self._simulation_step)

    def _initialize_robot_configuration(self):
        """Initialize robot configuration parameters."""
        self.initial_height = 1.15  # Initial height of the robot
        self.actuated_joint_home = np.zeros(32)
        self.actuated_joint_home[:6] = [0, 0, -0.44, 0.9, -0.45, 0]
        self.actuated_joint_home[6:12] = [0, 0, -0.44, 0.9, -0.45, 0]
        self.actuated_joint_home[14:22] = [0, -0.24, 0, -1, 0, 0, 0, 0]
        self.actuated_joint_home[22:30] = [0, -0.24, 0, -1, 0, 0, 0, 0]
        self.home_configuration = np.hstack(
            [np.array([0, 0, self.initial_height, 0, 0, 0, 1]), self.actuated_joint_home]
        )

    def _initialize_simulator(self):
        """Initialize simulator and robot."""
        self.robot_wrapper = pin.RobotWrapper.BuildFromURDF(
            self.urdf_path, self.mesh_path, pin.JointModelFreeFlyer(), True, None
        )
        self.robot_model = self.robot_wrapper.model
        self.simulator = PybulletWrapper(sim_rate=1000)
        self.robot = Robot(
            self.simulator,
            self.urdf_path,
            self.robot_model,
            [0, 0, self.initial_height],
            [0, 0, 0, 1],
            q=self.home_configuration,
            useFixedBase=False,
            verbose=True,
        )
        pb.resetDebugVisualizerCamera(
            cameraDistance=1.2, cameraYaw=90, cameraPitch=-20, cameraTargetPosition=[0.0, 0.0, 0.8]
        )

    def _setup_controller(self):
        """Setup PD controller gains."""
        self.Kp_matrix = np.eye(32)
        self.Kd_matrix = np.eye(32)

        # Set gains for different robot segments
        for i in range(0, 12):  # Legs
            self.Kp_matrix[i, i] = 930
            self.Kd_matrix[i, i] = 1.0 
        for i in range(12, 14):  # Torso
            self.Kp_matrix[i, i] = 580
            self.Kd_matrix[i, i] = 1.5
        for i in range(14, 30):  # Arms
            self.Kp_matrix[i, i] = 280
            self.Kd_matrix[i, i] = 0.5
        for i in range(30, 32):  # Head
            self.Kd_matrix[i, i] = 0.5

    def _setup_trajectory(self):
        """Setup trajectory for the robot."""
        self.initial_configuration = self.robot._q
        self.trajectory = self._generate_spline_trajectory(
            self.robot, self.robot_model, self.robot_model.createData(),
            self.initial_configuration, self.home_configuration,
            self.trajectory_duration, self.simulation_timestep
        )
        self.trajectory_index = 0
        self.trajectory_active = True

    def _generate_spline_trajectory(self, robot, model, data, q_start, q_end, duration, timestep):
        """Generate spline trajectory."""
        steps = int(duration / timestep)
        trajectory = []
        for i in range(steps + 1):
            t = min(1.0, i / steps)
            q_i = pin.interpolate(model, q_start, q_end, t)
            trajectory.append(q_i.copy())
        return trajectory

    def _publish_joint_states(self, q, v, tau):
        """Publish joint states."""
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = self.robot.actuatedJointNames()
        joint_state_msg.position = q.tolist()
        joint_state_msg.velocity = v.tolist()
        joint_state_msg.effort = tau.tolist()
        self.joint_state_publisher.publish(joint_state_msg)

    def _simulation_step(self):
        """Perform a simulation step."""
        if not rclpy.ok():
            return

        self.simulator.step()
        self.robot.update()

        if self.trajectory_active and self.trajectory_index < len(self.trajectory):
            q_desired_full = self.trajectory[self.trajectory_index]
            q_desired = q_desired_full[7:]
            self.trajectory_index += 1
            if self.trajectory_index >= len(self.trajectory):
                self.trajectory_active = False
                self.get_logger().info("Trajectory completed!")
        else:
            q_desired = self.home_configuration[7:]

        q_current = self.robot._q[7:]
        v_current = self.robot._v[6:]
        position_error = q_desired - q_current
        tau = self.Kp_matrix @ position_error - self.Kd_matrix @ v_current
        self.robot.setActuatedJointTorques(tau)

        self.publish_counter += 1
        if self.publish_counter >= self.publish_interval:
            self._publish_joint_states(q_current, v_current, tau)
            self.publish_counter = 0


def main(args=None):
    """Main function to start the simulation."""
    rclpy.init(args=args)
    simulator = TalosSimulator()
    try:
        rclpy.spin(simulator)
    except KeyboardInterrupt:
        pass
    finally:
        simulator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()