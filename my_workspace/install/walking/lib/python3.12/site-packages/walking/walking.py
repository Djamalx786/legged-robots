"""
talos walking simulation
"""

import numpy as np
import pinocchio as pin

import rclpy
from rclpy.node import Node

# simulator
from simulator.pybullet_wrapper import PybulletWrapper

# robot configs
import walking.talos_conf as conf

# modules
from walking.footstep_planner import FootStepPlanner, Side
from walking.lip_mpc import LIPMPC, LIPInterpolator, generate_zmp_reference
        
################################################################################
# main
################################################################################  
    
def main(): 
    
    ############################################################################
    # setup
    ############################################################################
    
    # setup ros
    rclpy.init()
    node = Node("walking_simulation")

    # setup the simulator
    simulator = PybulletWrapper(sim_rate=1000)

    # setup the robot
    robot = pin.RobotWrapper.BuildFromURDF(conf.robot_urdf_path, [conf.robot_mesh_path])

    # initial footsteps
    T_swing_w = conf.T_left_foot_w
    T_support_w = conf.T_right_foot_w

    # setup the plan with 20 steps
    no_steps = 20
    planner = FootStepPlanner(conf)
    plan = planner.planLine(T_support_w, Side.RIGHT, no_steps)
    plan.append(plan[-2])  # Append last two steps again
    plan.append(plan[-1])

    # generate reference
    ZMP_ref = generate_zmp_reference(np.array([step.pose.translation[:2] for step in plan]), conf.no_samples_per_step)
    planner.plot(simulator)

    # setup the lip models
    mpc = LIPMPC(conf)

    # Assume the com is over the first support foot
    x0 = np.zeros(4)
    interpolator = LIPInterpolator(x0, conf)

    # set the com task reference to the initial support foot
    interpolator.x = np.array([T_support_w.translation[0], 0.0, T_support_w.translation[1], 0.0])
    # Wait for the COM to shift
    pre_dur = 3.0
    N_pre = int(pre_dur / conf.dt)
    for _ in range(N_pre):
        simulator.step()
        robot.forwardKinematics()
    
    ############################################################################
    # logging
    ############################################################################

    pre_dur = 3.0   # Time to wait befor walking should start
    
    # Compute number of iterations:
    N_pre = int(pre_dur / conf.dt)  # Number of simulation steps before walking starts
    N_sim = int(conf.sim_duration / conf.dt)  # Total number of simulation steps during walking
    N_mpc = int(N_sim / conf.no_sim_per_mpc)  # Total number of MPC steps during walking
    
    # Create vectors to log all the data of the simulation
    COM_POS = np.nan * np.empty((N_sim, 3))  # Center of mass position
    COM_VEL = np.nan * np.empty((N_sim, 3))  # Center of mass velocity
    COM_ACC = np.nan * np.empty((N_sim, 3))  # Center of mass acceleration
    ZMP = np.nan * np.empty((N_sim, 2))  # Zero Moment Point
    DCM = np.nan * np.empty((N_sim, 2))  # Divergent Component of Motion
    TIME = np.nan * np.empty(N_sim)  # Simulation time
    
    ############################################################################
    # logging
    ############################################################################
    
    k = 0                                               # current MPC index                          
    plan_idx = 1                                        # current index of the step within foot step plan
    t_step_elapsed = 0.0                                # elapsed time within current step (use to evaluate spline)
    t_publish = 0.0                                     # last publish time (last time we published something)
    
    for i in range(-N_pre, N_sim):
        t = simulator.time()
        dt = simulator.dt()

        ########################################################################
        # update the mpc every no_sim_per_mpc steps
        ########################################################################
        if i >= 0 and i % conf.no_sim_per_mpc == 0:
            # 1. Get the current LIP state x_k from the interpolator
            x_k = interpolator.x

            # 2. Extract the ZMP reference ZMP_ref_k over the current horizon
            ZMP_ref_k = ZMP_ref[k:k + conf.no_mpc_samples_per_horizon]

            # 3. Solve the mpc and get the first control u_k
            u_k = mpc.buildSolveOCP(x_k, ZMP_ref_k, plan_idx)

            # 4. Increment mpc counter k
            interpolator.integrate(u_k)
            k += 1

        ########################################################################
        # update the foot spline 
        ########################################################################
        if i >= 0 and t_step_elapsed >= conf.step_duration:
            # 1. Get the next step location for the swing foot from the plan
            next_step = plan[plan_idx]

            # 2. Set the swing foot of the robot depending on the side of the next step
            swing_foot_pose = next_step.pose
            if next_step.side == Side.LEFT:
                robot.data.oMi[conf.left_foot_id] = swing_foot_pose
            else:
                robot.data.oMi[conf.right_foot_id] = swing_foot_pose

            # 3. Set the support foot to the robot depending on the other side
            support_foot_pose = plan[plan_idx - 1].pose
            if next_step.side == Side.LEFT:
                robot.data.oMi[conf.right_foot_id] = support_foot_pose
            else:
                robot.data.oMi[conf.left_foot_id] = support_foot_pose

            # 4. Get the current location of the swing foot
            current_swing_foot_pose = robot.data.oMi[conf.left_foot_id if next_step.side == Side.LEFT else conf.right_foot_id]

            # 5. Plan a foot trajectory between current and next foot pose
            foot_trajectory = SwingFootTrajectory(current_swing_foot_pose, swing_foot_pose, conf.step_duration)

            # 6. Increment step counter plan_idx
            plan_idx += 1
            t_step_elapsed = 0.0

        ########################################################################
        # in every iteration when walking
        ########################################################################
        if i >= 0:
            # 1. Update the foot trajectory with current step time and set the new pose, velocity, and acceleration reference to the swing foot
            swing_pose, swing_vel, swing_acc = foot_trajectory.evaluate(t_step_elapsed)
            if plan[plan_idx].side == Side.LEFT:
                robot.data.oMi[conf.left_foot_id] = swing_pose
            else:
                robot.data.oMi[conf.right_foot_id] = swing_pose

            # 2. Update the interpolator with the latest command u_k computed by the MPC
            interpolator.integrate(u_k)

            # 3. Feed the COM tasks with the new COM reference position, velocity, and acceleration
            c, c_dot, c_ddot = interpolator.comState()

            # 4. Increment elapsed footstep time
            t_step_elapsed += dt

        ########################################################################
        # update the simulation
        ########################################################################
        simulator.step()
        robot.forwardKinematics()

        # publish to ros
        if t - t_publish > 1. / 30.:
            t_publish = t
            node.get_logger().info(f"Time: {t}")

        # store for visualizations
        if i >= 0:
            TIME[i] = t
            COM_POS[i], COM_VEL[i], COM_ACC[i] = interpolator.comState()
            ZMP[i] = interpolator.zmp()
            DCM[i] = interpolator.dcm()

    ########################################################################
    # enough with the simulation, lets plot
    ########################################################################
    
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-dark')

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(TIME, COM_POS[:, 0], label='COM X')
    axes[0].plot(TIME, COM_POS[:, 1], label='COM Y')
    axes[0].set_ylabel("COM Position (m)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(TIME, ZMP[:, 0], label='ZMP X')
    axes[1].plot(TIME, ZMP[:, 1], label='ZMP Y')
    axes[1].set_ylabel("ZMP Position (m)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(TIME, DCM[:, 0], label='DCM X')
    axes[2].plot(TIME, DCM[:, 1], label='DCM Y')
    axes[2].set_ylabel("DCM Position (m)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    # Initialize the ROS client library

    try:
        main()
    finally:
        # Shutdown the ROS client library
        rclpy.shutdown()