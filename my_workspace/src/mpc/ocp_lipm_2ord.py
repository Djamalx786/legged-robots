"""Task2: Linear inverted pendulum Trajectory planning

The goal of this file is to formulate the optimal control problem (OCP)
in equation 12. 

In this case we will solve the trajectory planning over the entire footstep plan
(= horizon) in one go.

Our state will be the position and velocity of the pendulum in the 2d plane.
x = [cx, vx, cy, vy]
And your control the ZMP position in the 2d plane
u = [px, py]

You will need to fill in the TODO to solve the task.
"""

import numpy as np

from pydrake.all import MathematicalProgram, Solve

import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')

################################################################################
# settings
################################################################################

# Robot Parameters:
# --------------

h           = 0.80   # fixed CoM height (assuming walking on a flat terrain)
g           = 9.81   # norm of the gravity vector
foot_length = 0.10   # foot size in the x-direction
foot_width  = 0.06   # foot size in the y-direciton

# OCP Parameters:
# --------------
T                     = 0.1                                # fixed sampling time interval of computing the ocp in [s]
STEP_TIME             = 0.8                                # fixed time needed for every foot step [s]

NO_SAMPLES_PER_STEP   = int(round(STEP_TIME/T))            # number of ocp samples per step

NO_STEPS              = 10                                 # total number of foot steps in the plan
TOTAL_NO_SAMPLES      = NO_SAMPLES_PER_STEP*NO_STEPS       # total number of ocp samples over the complete plan (= Horizon)

# Cost Parameters:
# ---------------
alpha       = 10**(-1)                                      # ZMP error squared cost weight (= tracking cost)
gamma       = 10**(-3)                                      # CoM velocity error squared cost weight (= smoothing cost)

################################################################################
# helper function for visualization and dynamics
################################################################################

def generate_foot_steps(foot_step_0, step_size_x, no_steps):
    """Write a function that generates footstep of step size = step_size_x in the 
    x direction starting from foot_step_0 located at (x0, y0).
    
    Args:
        foot_step_0 (_type_): first footstep position (x0, y0)
        step_size_x (_type_): step size in x direction
        no_steps (_type_): number of steps to take
    """

    #>>>>TODO: generate the foot step plan with no_steps
    #>>>>Hint: Check the pdf Fig.3 for inspiration
    
    foot_steps = np.zeros((no_steps, 2))
    
    # Extract initial position
    x0, y0 = foot_step_0
    
    for i in range(no_steps):
        # X position: advance by step_size_x for each step
        foot_steps[i, 0] = x0 + i * step_size_x
        
        # Y position: alternate between left and right foot
        # Assuming y0 is the center, alternate ±foot_width/2 from center
        if i % 2 == 0:  # Even steps (0, 2, 4, ...) - same side as initial
            foot_steps[i, 1] = y0
        else:  # Odd steps (1, 3, 5, ...) - opposite side
            foot_steps[i, 1] = -y0  # Flip to other side
    
    return foot_steps


def plot_foot_steps(foot_steps, XY_foot_print, ax):
    """Write a function that plots footsteps in the xy plane using the given
    footprint (length, width)
    You can use the function ax.fill() to genereate a colored rectanges.
    Color the left and right steps different and check if the step sequence makes sense.

    Args:
        foot_steps (_type_): the foot step plan
        XY_foot_print (_type_): the dimensions of the foot (x,y)
        ax (_type_): the axis to plot on
    """
    #>>>>TODO: Plot the the footsteps into ax 
    
    foot_length, foot_width = XY_foot_print
    
    for i, (x, y) in enumerate(foot_steps):
        # Calculate rectangle corners (centered at footstep position)
        x_min = x - foot_length / 2
        x_max = x + foot_length / 2
        y_min = y - foot_width / 2
        y_max = y + foot_width / 2
        
        # Define rectangle vertices
        vertices_x = [x_min, x_max, x_max, x_min, x_min]
        vertices_y = [y_min, y_min, y_max, y_max, y_min]
        
        # Color based on step index - alternate between left (red) and right (green)
        if i % 2 == 0:
            color = 'red'     # Left foot
            alpha = 0.7
        else:
            color = 'green'   # Right foot  
            alpha = 0.7
            
        # Plot the rectangular footprint
        ax.fill(vertices_x, vertices_y, color=color, alpha=alpha, edgecolor='black', linewidth=1)
        
        # Add step number annotation
        ax.text(x, y, str(i), ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('X Position [m]')
    ax.set_ylabel('Y Position [m]')
    ax.set_title('Footstep Plan')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    

def generate_zmp_reference(foot_steps, no_samples_per_step):
    """generate a function that computes a referecne trajecotry for the ZMP
    (We need this for the tracking cost in the cost function of eq. 12)
    Remember: Our goal is to keep the ZMP at the footstep center within each step.
    So for the # of samples a step is active the zmp_ref should be at that step.
    
    Returns a vector of size (TOTAL_NO_SAMPLES, 2)

    Args:
        foot_steps (_type_): the foot step plan
        no_samples_per_step (_type_): number of sampes per step
    """
    #>>>>TODO: Generate the ZMP reference based on given foot_steps
    
    no_steps = foot_steps.shape[0]
    total_samples = no_steps * no_samples_per_step
    
    # Initialize ZMP reference array
    zmp_ref = np.zeros((total_samples, 2))
    
    # For each footstep, set the ZMP reference to the footstep center
    # for all time samples during that step
    for step_idx in range(no_steps):
        # Calculate the sample indices for this step
        start_sample = step_idx * no_samples_per_step
        end_sample = (step_idx + 1) * no_samples_per_step
        
        # Set ZMP reference to the footstep position for all samples in this step
        zmp_ref[start_sample:end_sample, 0] = foot_steps[step_idx, 0]  # x position
        zmp_ref[start_sample:end_sample, 1] = foot_steps[step_idx, 1]  # y position
    
    return zmp_ref

################################################################################
# Dynamics of the simplified walking model
################################################################################

def continious_LIP_dynamics(g, h):
    """returns the matrices A,B of the continious LIP dynamics

    Args:
        g (_type_): gravity
        h (_type_): fixed height

    Returns:
        np.array: A, B
    """
    #>>>>TODO: Generate A, B for the continous linear inverted pendulum
    #>>>>Hint: Look at Eq. 4 and rewrite as a system first order diff. eq.
    
    # From equation 4: ẍ = (g/h) * x - (g/h) * p
    # Rewrite as first-order system: [x, ẋ]' = [ẋ, ẍ]'
    # State: [x, ẋ], Control: p
    # ẋ = v
    # v̇ = (g/h) * x - (g/h) * p
    
    A = np.array([[0, 1],
                  [g/h, 0]])
    
    B = np.array([[0],
                  [-g/h]])
    
    return A, B

def discrete_LIP_dynamics(delta_t, g, h):
    """returns the matrices static Ad,Bd of the discretized LIP dynamics

    Args:
        delta_t (_type_): discretization steps
        g (_type_): gravity
        h (_type_): height

    Returns:
        _type_: _description_
    """
    #>>>>TODO: Generate Ad, Bd for the discretized linear inverted pendulum
    
    # Get continuous dynamics
    A, B = continious_LIP_dynamics(g, h)
    
    # For the LIP system, we can compute the exact discretization
    # The eigenvalues of A are ±√(g/h), so we use the exact solution
    
    omega = np.sqrt(g/h)
    
    # Exact discretization for the LIP system
    Ad = np.array([[np.cosh(omega * delta_t), np.sinh(omega * delta_t) / omega],
                   [omega * np.sinh(omega * delta_t), np.cosh(omega * delta_t)]])
    
    Bd = np.array([[1 - np.cosh(omega * delta_t)],
                   [-omega * np.sinh(omega * delta_t)]]) / omega
    
    return Ad, Bd

################################################################################
# setup the plan references and system matrices
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
foot_steps = generate_foot_steps([0.0, -0.09], step_size, NO_STEPS)

# zmp reference trajecotry
#>>>>TODO: 2. generate the ZMP reference using generate_zmp_reference
zmp_ref = generate_zmp_reference(foot_steps, NO_SAMPLES_PER_STEP)

#>>>>Note: At this point you can already start plotting things to see if they
# really make sense!

# discrete LIP dynamics
#>>>>TODO: get the static dynamic matrix Ad, Bd
Ad_x, Bd_x = discrete_LIP_dynamics(T, g, h)
Ad_y, Bd_y = discrete_LIP_dynamics(T, g, h)  # Same for y-direction

# continous LIP dynamics
#>>>>TODO: get the static dynamic matrix A, B
A_x, B_x = continious_LIP_dynamics(g, h)
A_y, B_y = continious_LIP_dynamics(g, h)  # Same for y-direction

################################################################################
# problem definition
################################################################################

# Define an instance of MathematicalProgram 
prog = MathematicalProgram() 

################################################################################
# variables
nx = 4  #>>>>TODO: State dimension = 4 [cx, vx, cy, vy]
nu = 2  #>>>>TODO: control dimension = 2 [px, py]

state = prog.NewContinuousVariables(TOTAL_NO_SAMPLES, nx, 'state')
control = prog.NewContinuousVariables(TOTAL_NO_SAMPLES, nu, 'control')

# intial state
state_inital = np.array([x_0[0], x_0[1], y_0[0], y_0[1]])  #>>>>TODO: inital state if based on first footstep (+ zero velo)

# terminal state
state_terminal = np.array([foot_steps[-1, 0], 0.0, foot_steps[-1, 1], 0.0])  #>>>>TODO: terminal state if based on last footstep (+ zero velo)

################################################################################
# constraints

# 1. intial constraint
#>>>>TODO: Add inital state constrain, Hint: prog.AddConstraint
for i in range(nx):
    prog.AddConstraint(state[0, i] == state_inital[i])

# 2. terminal constraint
#>>>>TODO: Add terminal state constrain, Hint: prog.AddConstraint
for i in range(nx):
    prog.AddConstraint(state[TOTAL_NO_SAMPLES-1, i] == state_terminal[i])

# 3. at each step: respect the LIP descretized dynamics
#>>>>TODO: Enforce the dynamics at every time step
for k in range(TOTAL_NO_SAMPLES-1):
    # X-direction dynamics: [cx, vx]_{k+1} = Ad_x * [cx, vx]_k + Bd_x * px_k
    x_next = Ad_x @ state[k, [0, 1]] + Bd_x.flatten() * control[k, 0]
    prog.AddConstraint(state[k+1, 0] == x_next[0])  # cx position
    prog.AddConstraint(state[k+1, 1] == x_next[1])  # vx velocity
    
    # Y-direction dynamics: [cy, vy]_{k+1} = Ad_y * [cy, vy]_k + Bd_y * py_k
    y_next = Ad_y @ state[k, [2, 3]] + Bd_y.flatten() * control[k, 1]
    prog.AddConstraint(state[k+1, 2] == y_next[0])  # cy position
    prog.AddConstraint(state[k+1, 3] == y_next[1])  # vy velocity

# 4. at each step: keep the ZMP within the foot sole (use the footprint and planned step position)
#>>>>TODO: Add ZMP upper and lower bound to keep the control (ZMP) within each footprints
for k in range(TOTAL_NO_SAMPLES):
    # Compute footprint bounds around current ZMP reference
    px_min = zmp_ref[k, 0] - foot_length/2
    px_max = zmp_ref[k, 0] + foot_length/2
    py_min = zmp_ref[k, 1] - foot_width/2
    py_max = zmp_ref[k, 1] + foot_width/2
    
    # Add ZMP constraints
    prog.AddConstraint(control[k, 0] >= px_min)
    prog.AddConstraint(control[k, 0] <= px_max)
    prog.AddConstraint(control[k, 1] >= py_min)
    prog.AddConstraint(control[k, 1] <= py_max)

################################################################################
# stepwise cost, note that the cost function is scalar!

# setup our cost: minimize zmp error (tracking), minimize CoM velocity (smoothing)
#>>>>TODO: add the cost at each timestep, hint: prog.AddCost
for k in range(TOTAL_NO_SAMPLES):
    # ZMP tracking cost: alpha * ||p_k - p_ref_k||^2
    zmp_error = control[k, :] - zmp_ref[k, :]
    prog.AddCost(alpha * (zmp_error[0]**2 + zmp_error[1]**2))
    
    # CoM velocity smoothing cost: gamma * ||v_k||^2
    velocity = state[k, [1, 3]]  # [vx, vy]
    prog.AddCost(gamma * (velocity[0]**2 + velocity[1]**2))

################################################################################
# solve

result = Solve(prog)
if not result.is_success:
    print("failure")
print("solved")

# extract the solution
#>>>>TODO: extract your variables from the result object
state_opt = result.GetSolution(state)
control_opt = result.GetSolution(control)
t = T*np.arange(0, TOTAL_NO_SAMPLES)

# compute the acceleration
#>>>>TODO: compute the acceleration of the COM
# Using LIP dynamics: a = (g/h) * c - (g/h) * p
acc_x = (g/h) * state_opt[:, 0] - (g/h) * control_opt[:, 0]
acc_y = (g/h) * state_opt[:, 2] - (g/h) * control_opt[:, 1]

################################################################################
# plot something

#>>>>TODO: plot everything in x-axis
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# X-direction plots
axes[0, 0].plot(t, state_opt[:, 0], 'b-', linewidth=2, label='CoM Position X')
axes[0, 0].plot(t, control_opt[:, 0], 'r--', linewidth=2, label='ZMP Position X')
axes[0, 0].plot(t, zmp_ref[:, 0], 'g:', linewidth=2, label='ZMP Reference X')
axes[0, 0].set_ylabel('Position X [m]')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[1, 0].plot(t, state_opt[:, 1], 'b-', linewidth=2, label='CoM Velocity X')
axes[1, 0].set_ylabel('Velocity X [m/s]')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[2, 0].plot(t, acc_x, 'b-', linewidth=2, label='CoM Acceleration X')
axes[2, 0].set_ylabel('Acceleration X [m/s²]')
axes[2, 0].set_xlabel('Time [s]')
axes[2, 0].legend()
axes[2, 0].grid(True)

#>>>>TODO: plot everything in y-axis
# Y-direction plots
axes[0, 1].plot(t, state_opt[:, 2], 'b-', linewidth=2, label='CoM Position Y')
axes[0, 1].plot(t, control_opt[:, 1], 'r--', linewidth=2, label='ZMP Position Y')
axes[0, 1].plot(t, zmp_ref[:, 1], 'g:', linewidth=2, label='ZMP Reference Y')
axes[0, 1].set_ylabel('Position Y [m]')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 1].plot(t, state_opt[:, 3], 'b-', linewidth=2, label='CoM Velocity Y')
axes[1, 1].set_ylabel('Velocity Y [m/s]')
axes[1, 1].legend()
axes[1, 1].grid(True)

axes[2, 1].plot(t, acc_y, 'b-', linewidth=2, label='CoM Acceleration Y')
axes[2, 1].set_ylabel('Acceleration Y [m/s²]')
axes[2, 1].set_xlabel('Time [s]')
axes[2, 1].legend()
axes[2, 1].grid(True)

#>>>>TODO: plot everything in xy-plane
# XY-plane plot
plot_foot_steps(foot_steps, footprint, axes[0, 2])
axes[0, 2].plot(state_opt[:, 0], state_opt[:, 2], 'b-', linewidth=3, label='CoM Trajectory')
axes[0, 2].plot(control_opt[:, 0], control_opt[:, 1], 'r--', linewidth=2, label='ZMP Trajectory')
axes[0, 2].plot(zmp_ref[:, 0], zmp_ref[:, 1], 'g:', linewidth=2, label='ZMP Reference')
axes[0, 2].plot(state_opt[0, 0], state_opt[0, 2], 'go', markersize=8, label='Start')
axes[0, 2].plot(state_opt[-1, 0], state_opt[-1, 2], 'ro', markersize=8, label='End')
axes[0, 2].legend()
axes[0, 2].set_title('Walking Pattern - Top View')

# Remove empty subplots
axes[1, 2].remove()
axes[2, 2].remove()

plt.tight_layout()
plt.show()