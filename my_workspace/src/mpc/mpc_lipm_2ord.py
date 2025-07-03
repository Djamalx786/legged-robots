"""Task2: Linear inverted pendulum MPC

The goal of this file is to formulate the optimal control problem (OCP)
in equation 12 but this time as a model predictive controller (MPC).

In this case we will solve the trajectory planning multiple times over 
a shorter horizon of just 2 steps (receding horizon).
Time between two MPC updates is called T_MPC.

In between MPC updates we simulate the Linear inverted pendulum at a smaller
step time T_SIM, with the lates MPC control ouput u.

Our state & control is the same as before
x = [cx, vx, cy, vy]
u = [px, py]

You will need to fill in the TODO to solve the task.
"""

import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import MathematicalProgram, Solve, eq

import matplotlib.animation as animation

################################################################################
# settings
################################################################################

NO_STEPS                = 8         # total number of foot steps
STEP_TIME               = 0.8       # time needed for every step

# Robot Parameters:
# --------------
h                       = 0.80      # fixed CoM height (assuming walking on a flat terrain)
g                       = 9.81      # norm of the gravity vector
foot_length             = 0.10      # foot size in the x-direction
foot_width              = 0.06      # foot size in the y-direciton


# MPC Parameters:
# --------------
T_MPC                   = 0.1                                               # sampling time interval of the MPC
NO_MPC_SAMPLES_PER_STEP = int(round(STEP_TIME/T_MPC))                       # number of mpc updates per step

NO_STEPS_PER_HORIZON  = 2                                                   # how many steps in the horizon
T_HORIZON = NO_STEPS_PER_HORIZON*STEP_TIME                                  # duration of future horizon
NO_MPC_SAMPLES_HORIZON = int(round(NO_STEPS_PER_HORIZON*STEP_TIME/T_MPC))   # number of mpc updates per horizon

# Cost Parameters:
# ---------------
alpha       = 10**(-1)                                  # ZMP error squared cost weight (= tracking cost)
gamma       = 10**(-3)                                  # CoM velocity error squared cost weight (= smoothing cost)

# Simulation Parameters:
# --------------
T_SIM                   = 0.005                         # 200 Hz simulation time

NO_SIM_SAMPLES_PER_MPC = int(round(T_MPC/T_SIM))        # NO SIM samples between MPC updates
NO_MPC_SAMPLES = int(round(NO_STEPS*STEP_TIME/T_MPC))   # Total number of MPC samples
NO_SIM_SAMPLES = int(round(NO_STEPS*STEP_TIME/T_SIM))   # Total number of Simulator samples

################################################################################
# Helper fnc
################################################################################

def generate_foot_steps(foot_step_0, step_size_x, no_steps):
    """Write a function that generates footstep of stepsize=step_size_x in the 
    x direction starting from foot_step_0 located at (x0, y0).

    Args:
        foot_step_0 (_type_): _description_
        step_size_x (_type_): _description_
        no_steps (_type_): _description_
    """

    #>>>>TODO: copy from previous Task 2
    
    foot_steps = np.zeros((no_steps, 2)) 
    foot_steps[0] = foot_step_0 
    for i in range(1, no_steps):
        
        foot_steps[i, 0] = foot_steps[i-1, 0] + step_size_x
        foot_steps[i, 1] = foot_steps[i-1, 1] - step_size_x * (-1)**i  

    return foot_steps


def plot_foot_steps(foot_steps, XY_foot_print, ax):
    """Write a function that plots footsteps in the xy plane using the given
    footprint (length, width)
    You can use the function ax.fill() to gerneate a rectable.
    Color left and right steps differt and check if the step sequence makes sense.

    Args:
        foot_steps (_type_): _description_
    """
    #>>>>TODO: copy from previous Task 2
    foot_length, foot_width = XY_foot_print
    for i, step in enumerate(foot_steps):
        center_x, center_y = step
        x_corners = [center_x - foot_length / 2, center_x + foot_length / 2,
                     center_x + foot_length / 2, center_x - foot_length / 2]
        y_corners = [center_y - foot_width / 2, center_y - foot_width / 2,
                     center_y + foot_width / 2, center_y + foot_width / 2]
        color = 'blue' if i % 2 == 0 else 'orange'  
        ax.fill(x_corners, y_corners, color=color, alpha=0.5, edgecolor='black', linewidth=1)


    ax.legend()
    ax.set_xlabel('X Position [m]')
    ax.set_ylabel('Y Position [m]')
    ax.set_title('Footstep Plan')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

def generate_zmp_reference(foot_steps, no_samples_per_step):
    """generate a function that computes a referecne trajecotry for the zmp.
    Our goal is to keep the ZMP at the footstep center within each step

    Args:
        foot_steps (_type_): _description_
        no_samples_per_step (_type_): _description_
    """
    #>>>>TODO: copy from previous Task 2
    total_samples = len(foot_steps) * no_samples_per_step
    zmp_ref = np.zeros((total_samples, 2))
    for i, step in enumerate(foot_steps):
        start = i * no_samples_per_step
        end = start + no_samples_per_step
        zmp_ref[start:end, 0] = step[0] 
        zmp_ref[start:end, 1] = step[1]  
        
    return zmp_ref

################################################################################
# Dynamics of the simplified walking model
################################################################################

def continious_LIP_dynamics():
    """returns the static matrices A,B of the continious LIP dynamics

    Args:
        g (_type_): gravity
        h (_type_): height

    Returns:
        np.array: A, B
    """

    #>>>>TODO: copy from previous Task 2
    A = np.array([[0,   1,   0,   0],
                  [g/h, 0,   0,   0],
                  [0,   0,   0,   1],
                  [0,   0,   g/h, 0]])
    
    B = np.array([[0,    0],
                  [-g/h, 0],
                  [0,    0],
                  [0,    -g/h]])
    
    return A, B

def discrete_LIP_dynamics(delta_t, g, h):
    """returns the matrices static Ad,Bd of the discretized LIP dynamics

    Args:
        dt (_type_): discretization steps
        g (_type_): gravity
        h (_type_): height

    Returns:
        _type_: _description_
    """
    #>>>>TODO: copy from previous Task 2
    omega = np.sqrt(g/h)
    Ad_xy = np.array([[np.cosh(omega*delta_t), np.sinh(omega*delta_t)/omega],
                   [omega*np.sinh(omega*delta_t), np.cosh(omega*delta_t)]])
    Bd_xy = np.array([1-np.cosh(omega*delta_t), -omega *np.sinh(omega*delta_t)])
    
    Ad = np.zeros((4, 4))
    Ad[0:2, 0:2] = Ad_xy
    Ad[2:4, 2:4] = Ad_xy
    
    Bd = np.zeros((4, 2))
    Bd[0:2, 0] = Bd_xy.flatten()
    Bd[2:4, 1] = Bd_xy.flatten()
    
    return Ad, Bd

################################################################################
# Simulation
################################################################################

class Simulator:
    """Simulates the Linear inverted pendulum continous dynamics
    Uses simple euler integration to simulate LIP at sample time dt
    """
    def __init__(self, x_inital, dt):
        self.dt = dt
        self.x = x_inital
        
        self.A, self.B = continious_LIP_dynamics()
        self.D = np.array([[0, 0], [1, 0], [0, 0], [0, 1]])
        
    def simulate(self, u, d=np.zeros(2)):
        """updates the LIP state x using based on command u
        
        Optionally: Takes a disturbance acceleration d to simulate effect
        of external pushes on the LIP.
        """

        #>>>>TODO: Compute x_dot and use euler integration to approximate
        # the state at t+dt
        #>>>>TODO: The disturbance is added in x_dot as self.D@d
        x_dot = self.A @ self.x + self.B @ u + self.D @ d
        self.x = self.x + self.dt * x_dot
        
        return self.x    

################################################################################
# MPC
################################################################################

class MPC:
    """MPC for the Linear inverted pendulum
    """
    def __init__(self, dt, T_horizon):
        self.dt = dt                                        # mpc dt
        self.T_horizon = T_horizon                          # time of horizon
        self.no_samples = int(round(T_horizon/self.dt))     # mpc samples in horizon (nodes)

        self.Ad, self.Bd = discrete_LIP_dynamics(dt, g, h)
        
        self.X_k = None                                     # state over current horizon
        self.U_k = None                                     # control over current horizon
        self.ZMP_ref_k = None                               # ZMP reference over current horizon
        
    def buildSolveOCP(self, x_k, ZMP_ref_k, terminal_idx):
        """ build the MathematicalProgram that solves the mpc problem and 
        returns the first command of U_k

        Args:
            x_k (_type_): the current state of the lip when starting the mpc
            ZMP_ref_k (_type_): the reference over the current horizon, shape=(no_samples, 2)
            terminal_idx (_type_): index of the terminal constraint within horizon (or bigger than horizon if no constraint)
            
        """
        
        # variables
        nx = 4  # State dimension = [cx, vx, cy, vy]
        nu = 2  # Control dimension = [px, py]
        prog = MathematicalProgram()
        
        state = prog.NewContinuousVariables(self.no_samples, nx, 'state')
        control = prog.NewContinuousVariables(self.no_samples, nu, 'control')
        
        # 1. intial constraint
        #>>>>TODO: Add inital state constraint, Hint: x_k
        prog.AddConstraint(eq(state[0, :], x_k))

        # 2. at each time step: respect the LIP descretized dynamics
        #>>>>TODO: Enforce the dynamics at every time step
        for k in range(self.no_samples - 1):
            prog.AddConstraint(eq(state[k + 1, :], self.Ad @ state[k, :] + self.Bd @ control[k, :]))

        # 3. at each time step: keep the ZMP within the foot sole (use the footprint and planned step position)
        #>>>>TODO: Add ZMP upper and lower bound to keep the control (ZMP) within each footprints
        #Hint: first compute upper and lower bound based on zmp_ref then add constraints.
        #Hint: Add constraints at every time step
        zmp_lb = ZMP_ref_k - footprint/2
        zmp_ub = ZMP_ref_k + footprint/2
        for k in range(self.no_samples):
            prog.AddConstraint(control[k, 0] >= zmp_lb[k, 0])
            prog.AddConstraint(control[k, 0] <= zmp_ub[k, 0])
            prog.AddConstraint(control[k, 1] >= zmp_lb[k, 1])
            prog.AddConstraint(control[k, 1] <= zmp_ub[k, 1])

    
        # 4. if terminal_idx < self.no_samples than we have the terminal state within
        # the current horizon. In this case create the terminal state (foot step pos + zero vel)
        # and apply the state constraint to all states >= terminal_idx within the horizon
        #>>>>TODO: Add the terminal constraint if requires
        #Hint: If you are unsure, you can start testing without this first!

        if terminal_idx < self.no_samples:
            # The terminal state is the center of the footstep with zero velocity
            # terminal_state = np.concatenate([ZMP_ref_k[terminal_idx], [0, 0]])
            # prog.AddConstraint(eq(state[terminal_idx, :], terminal_state))
            terminal_state = np.array([ZMP_ref_k[terminal_idx, 0], 0, ZMP_ref_k[terminal_idx, 1], 0])
            for k_term in range(terminal_idx, self.no_samples):
                prog.AddConstraint(eq(state[k_term, :], terminal_state))

        # setup our cost: minimize zmp error (tracking), minimize CoM velocity (smoothing)
        #>>>>TODO: add the cost at each timestep, hint: prog.AddCost
        for k in range(self.no_samples):
            # ZMP error cost
            zmp_error = control[k,:] - ZMP_ref_k[k,:]
            prog.AddCost(alpha * zmp_error.dot(zmp_error))
            
            # CoM velocity cost
            com_vel = state[k, [1,3]]
            prog.AddCost(gamma * com_vel.dot(com_vel))
            
        # solve
        result = Solve(prog)
        if not result.is_success:
            print("failure")
            
        self.X_k = result.GetSolution(state)
        self.U_k = result.GetSolution(control)
        if np.isnan(self.X_k).any():
            print("failure")
        
        self.ZMP_ref_k = ZMP_ref_k
        return self.U_k[0]
    
################################################################################
# run the simulation
################################################################################

# inital state in x0 = [px0, vx0]
x_0 = np.array([0.0, 0.0])
# inital state in y0 = [py0, vy0]
y_0 = np.array([-0.09, 0.0])

# footprint
footprint = np.array([foot_length, foot_width])

# generate the footsteps
step_size = 0.2
#>>>>TODO: 1. generate the foot step plan using generate_foot_steps
first_foot_step = np.array([x_0[0], y_0[0]])
foot_steps = generate_foot_steps(first_foot_step, step_size, NO_STEPS)


# reapeat the last two foot steps (so the mpc horizon never exceeds the plan!)
foot_steps = np.vstack([
    foot_steps, foot_steps[-1], foot_steps[-1]])

# zmp reference trajecotry
#>>>>TODO: 2. generate the complete ZMP reference using generate_zmp_reference
ZMP_ref = generate_zmp_reference(foot_steps, NO_MPC_SAMPLES_PER_STEP)

# generate mpc
mpc = MPC(T_MPC, T_HORIZON)

# generate the pendulum simulator
state_0 = np.concatenate([x_0, y_0])
sim = Simulator(state_0, T_SIM)

# setup some vectors for plotting stuff
TIME_VEC = np.nan*np.ones(NO_SIM_SAMPLES)
STATE_VEC = np.nan*np.ones([NO_SIM_SAMPLES, 4])
ZMP_REF_VEC = np.nan*np.ones([NO_SIM_SAMPLES, 2])
ZMP_VEC = np.nan*np.ones([NO_SIM_SAMPLES, 2])

# time to add some disturbance
t_push = 3.2

# execution loop
u_k = np.zeros(2)
k = 0   # the number of mpc update
for i in range(NO_SIM_SAMPLES):
    
    # simulation time
    t = i*T_SIM
        
    if i % NO_SIM_SAMPLES_PER_MPC == 0:
        # time to update the mpc
        
        # current state
        #>>>>TODO: get current state from the simulator
        x_k = sim.x
    
        #>>>>TODO: extract the current horizon from the complete reference trajecotry ZMP_ref
        # ZMP_ref_k = ZMP_ref[k:k+mpc.no_samples]
        #>>>>TODO: extract the current horizon from the complete reference trajecotry ZMP_ref
        end_idx = min(k + mpc.no_samples, len(ZMP_ref))
        ZMP_ref_k = ZMP_ref[k:end_idx]
        
        # If we don't have enough samples, pad with the last available reference
        if len(ZMP_ref_k) < mpc.no_samples:
            last_ref = ZMP_ref_k[-1] if len(ZMP_ref_k) > 0 else ZMP_ref[-1]
            padding = np.tile(last_ref, (mpc.no_samples - len(ZMP_ref_k), 1))
            ZMP_ref_k = np.vstack([ZMP_ref_k, padding])
    
        real_samples_in_horizon = len(ZMP_ref) - k
        if real_samples_in_horizon < mpc.no_samples:
            # Terminal constraint should start where real data ends
            idx_terminal_k = real_samples_in_horizon
        else:
            idx_terminal_k = mpc.no_samples + 1
        
        #>>>>TODO: Update the mpc, get new command
        u_k = mpc.buildSolveOCP(x_k, ZMP_ref_k, idx_terminal_k)
        
        k += 1
    
    # simulate a push for 0.05 sec with 1.0 m/s^2 acceleration 
    x_ddot_ext = np.array([0, 0])
    
    # #>>>>TODO: when you got everything working try adding a small disturbance
    if i > int(t_push/T_SIM) and i < int((t_push + 0.05)/T_SIM):
       x_ddot_ext = np.array([0, 1.0])
    
    #>>>>TODO: Update the simulation using the current command
    x_k = sim.simulate(u_k, d=x_ddot_ext)
    
    # save some stuff
    TIME_VEC[i] = t
    STATE_VEC[i] = x_k
    ZMP_VEC[i] = u_k
    if mpc.ZMP_ref_k is not None:
        ZMP_REF_VEC[i] = mpc.ZMP_ref_k[0]
    
ZMP_LB_VEC = ZMP_REF_VEC - footprint[None,:]/2
ZMP_UB_VEC = ZMP_REF_VEC + footprint[None,:]/2

#>>>>TODO: Use the recodings in STATE_VEC and ZMP_VEC to compute the 
# LIP acceleration
#>>>>Hint: Use the continious dynamic matrices
A, B = continious_LIP_dynamics()
STATE_DOT_VEC = (A @ STATE_VEC.T + B @ ZMP_VEC.T).T


################################################################################
# plot something

#>>>>TODO: plot everything in x-axis
fig, ax = plt.subplots(3,1, sharex=True, figsize=(8,10))
ax[0].plot(TIME_VEC, STATE_VEC[:,0], label='CoM Pos')
ax[0].plot(TIME_VEC, ZMP_VEC[:,0], label='ZMP')
ax[0].plot(TIME_VEC, ZMP_REF_VEC[:,0], '--', label='ZMP Ref')
ax[0].plot(TIME_VEC, ZMP_LB_VEC[:,0], 'r--', label='ZMP bounds')
ax[0].plot(TIME_VEC, ZMP_UB_VEC[:,0], 'r--')
ax[0].axvspan(t_push, t_push+0.05, color='red', alpha=0.2, label='Disturbance')
ax[0].set_ylabel('position [m]')
ax[0].legend()

ax[1].plot(TIME_VEC, STATE_VEC[:,1], label='CoM Vel')
ax[1].set_ylabel('velocity [m/s]')
ax[1].legend()

ax[2].plot(TIME_VEC, STATE_DOT_VEC[:,1], label='CoM Acc')
ax[2].set_ylabel('acceleration [m/s^2]')
ax[2].set_xlabel('time [s]')
ax[2].legend()
fig.suptitle('X-axis Trajectories')
fig.savefig('lipm_mpc_x_axis.png', dpi=300)


#>>>>TODO: plot everything in y-axis
fig, ax = plt.subplots(3,1, sharex=True, figsize=(8,10))
ax[0].plot(TIME_VEC, STATE_VEC[:,2], label='CoM Pos')
ax[0].plot(TIME_VEC, ZMP_VEC[:,1], label='ZMP')
ax[0].plot(TIME_VEC, ZMP_REF_VEC[:,1], '--', label='ZMP Ref')
ax[0].plot(TIME_VEC, ZMP_LB_VEC[:,1], 'r--', label='ZMP bounds')
ax[0].plot(TIME_VEC, ZMP_UB_VEC[:,1], 'r--')
ax[0].axvspan(t_push, t_push+0.05, color='red', alpha=0.2, label='Disturbance')
ax[0].set_ylabel('position [m]')
ax[0].legend()

ax[1].plot(TIME_VEC, STATE_VEC[:,3], label='CoM Vel')
ax[1].set_ylabel('velocity [m/s]')
ax[1].legend()

ax[2].plot(TIME_VEC, STATE_DOT_VEC[:,3], label='CoM Acc')
ax[2].set_ylabel('acceleration [m/s^2]')
ax[2].set_xlabel('time [s]')
ax[2].legend()
fig.suptitle('Y-axis Trajectories')
fig.savefig('lipm_mpc_y_axis.png', dpi=300)


#>>>>TODO: plot everything in xy-plane
fig, ax = plt.subplots(figsize=(8, 8))
plot_foot_steps(foot_steps, footprint, ax)
ax.plot(ZMP_ref[:,0], ZMP_ref[:,1], '-o', label='ZMP Reference')
ax.plot(STATE_VEC[:,0], STATE_VEC[:,2], '-', label='CoM Trajectory')
ax.plot(ZMP_VEC[:,0], ZMP_VEC[:,1], '-', label='ZMP Trajectory')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.legend()
ax.set_title('XY Plane Trajectories')
ax.set_aspect('equal')
fig.savefig('lipm_mpc_xy_plane.png', dpi=300)
plt.show()