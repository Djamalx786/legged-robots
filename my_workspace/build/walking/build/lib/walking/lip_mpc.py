import numpy as np

from pydrake.all import MathematicalProgram, Solve, eq

################################################################################
# Helper fnc
################################################################################

def continious_LIP_dynamics(g, h):
    """returns the static matrices A,B of the continious LIP dynamics
    """
    #>>>>TODO: Compute
    # create 1d matrix A and B
    A_1 = np.zeros((2,2))
    A_1[1,0] = g/h
    A_1[0,1] = 1

    B_1 = np.zeros((2,))
    B_1[1] = - g/h

    # create matrix A
    A = np.zeros((4,4))
    A[:2,:2] = A_1
    A[2:,2:] = A_1

    # create matrix B
    B = np.zeros((4,2))
    B[:2,0] = B_1
    B[2:,1] = B_1

    return A, B

def discrete_LIP_dynamics(g, h, dt):
    """returns the matrices static Ad,Bd of the discretized LIP dynamics
    """
    #>>>>TODO: Compute
    # create omega
    omega = np.sqrt(g/h)

    # create 1d matrix Ad
    Ad_1 = np.zeros((2,2))
    Ad_1[0,0] = np.cosh( omega * dt )
    Ad_1[1,1] = np.cosh( omega * dt )
    Ad_1[1,0] = omega * np.sinh( omega * dt )
    Ad_1[0,1] = (1/omega) * np.sinh( omega * dt )

    # create 1d matrix Bd
    Bd_1 = np.zeros((2,))
    Bd_1[0] = 1-np.cosh( omega * dt )
    Bd_1[1] = - omega * np.sinh( omega * dt )

    # create Ad
    Ad = np.zeros((4,4))
    Ad[:2,:2] = Ad_1
    Ad[2:,2:] = Ad_1

    # create Bd
    Bd = np.zeros((4,2))
    Bd[:2,0] = Bd_1
    Bd[2:,1] = Bd_1

    return Ad, Bd

################################################################################
# LIPInterpolator
################################################################################

class LIPInterpolator:
    """Integrates the linear inverted pendulum model using the 
    continous dynamics. To interpolate the solution to hight 
    """
    def __init__(self, x_inital, conf):
        self.conf = conf
        self.dt = conf.dt
        self.x = x_inital
        self.A, self.B = continious_LIP_dynamics(self.conf.g, self.conf.h)
        self.x_dot = self.A @ self.x
        
    def integrate(self, u):
        #>>>>TODO: integrate with dt
        self.x_dot = self.A @ self.x + self.B @ u
        self.x = self.x +self.x_dot * self.dt

        return self.x
    
    def comState(self):
        #>>>>TODO: return the center of mass state
        # that is position \in R3, velocity \in R3, acceleration \in R3
        c = np.array([self.x[0], self.x[2], self.conf.h])
        c_dot = np.array([self.x[1], self.x[3], 0.0])
        c_ddot = np.array([self.x_dot[1], self.x_dot[3], 0.0])
        return c, c_dot, c_ddot
    
    def dcm(self):
        #>>>>TODO: return the computed dcm
        # DCM = CoM + c_dot / omega

        omega = np.sqrt(self.conf.g/self.conf.h)

        dcm = np.array([self.x[0], self.x[2], self.conf.h]) + np.array([self.x[1], self.x[3], 0.0]) / omega
        return dcm[:2]
    
    def zmp(self):
        #>>>>TODO: return the zmp
        # ZMP = CoM - (h/g)*c_ddot
        zmp = np.array([self.x[0], self.x[2], self.conf.h]) - (self.conf.h/self.conf.g) * np.array([self.x_dot[1], self.x_dot[3], 0.0])
        return zmp[:2]
        
    
################################################################################
# LIPMPC
################################################################################

class LIPMPC:
    def __init__(self, conf):
        self.conf = conf
        self.dt = conf.dt        
        self.no_samples = conf.no_mpc_samples_per_horizon
        self.Ad, self.Bd = discrete_LIP_dynamics(self.conf.g, self.conf.h, self.dt)
        
        # solution and references over the horizon
        self.X_k = None
        self.U_k = None
        self.ZMP_ref_k = None
        
    def buildSolveOCP(self, x_k, ZMP_ref_k, terminal_idx):
        """build and solve ocp

        Args:
            x_k (_type_): inital mpc state
            ZMP_ref_k (_type_): zmp reference over horizon
            terminal_idx (_type_): index within horizon to apply terminal constraint

        Returns:
            _type_: control
        """
        
        #>>>>TODO: build and solve the ocp
        #>>>>Note: start without terminal constraints
        
        # variables
        nx = 4 
        nu = 2 
        prog = MathematicalProgram()
        
        state = prog.NewContinuousVariables(self.no_samples, nx, 'state')
        control = prog.NewContinuousVariables(self.no_samples, nu, 'control')
        
        # 1. intial constraint
        prog.AddConstraint(eq(state[0, :],x_k))

        # 2. at each time step: respect the LIP descretized dynamics
        for k in range(self.no_samples - 1):
            prog.AddConstraint(eq(state[k+1,:], self.Ad @ state[k,:] + self.Bd @ control[k,:]))
        
        # get foot_length and foot_width
        foot_length = self.conf.lfxp + self.conf.lfxn
        foot_width = self.conf.lfyp + self.conf.lfyn

        # 3. at each time step: keep the ZMP within the foot sole (use the footprint and planned step position)
        for k in range(self.no_samples):
            prog.AddBoundingBoxConstraint(
                ZMP_ref_k[k, 0] - foot_length/2,
                ZMP_ref_k[k, 0] + foot_length/2,
                control[k, 0]
            )
            prog.AddBoundingBoxConstraint(
                ZMP_ref_k[k, 1] - foot_width/2,
                ZMP_ref_k[k, 1] + foot_width/2,
                control[k, 1]
            )
    
        # 4. if terminal_idx < self.no_samples than we have the terminal state within
        # the current horizon. In this case create the terminal state (foot step pos + zero vel)
        # and apply the state constraint to all states >= terminal_idx within the horizon
        if (terminal_idx < self.no_samples):
            terminal_state = np.array([
                ZMP_ref_k[terminal_idx, 0], 0.0,
                ZMP_ref_k[terminal_idx, 1], 0.0
            ])
            for k in range(terminal_idx, self.no_samples):
                prog.AddConstraint(eq(state[k,:], terminal_state[:]))
    
        # setup our cost: minimize zmp error (tracking), minimize CoM velocity (smoothing)
        for k in range(self.no_samples):
            prog.AddCost(self.conf.alpha * ((control[k, 0] - ZMP_ref_k[k, 0])**2 + (control[k, 1] - ZMP_ref_k[k, 1])**2))
            prog.AddCost(self.conf.gamma * (state[k, 1]**2 + state[k, 3]**2))

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
    

def generate_zmp_reference(foot_steps, no_samples_per_step):
    """generate the zmp reference given a sequence of footsteps
    """

    #>>>>TODO: use the previously footstep type to build the reference 
    # trajectory for the zmp

    zmp_ref = np.zeros((foot_steps.shape[0] * no_samples_per_step, 2))

    for step in range(foot_steps.shape[0]):
        start_idx = step * no_samples_per_step
        end_idx = (step + 1) * no_samples_per_step
        zmp_ref[start_idx:end_idx, :] = foot_steps[step,:].reshape(1, 2)

    return zmp_ref
