import numpy as np
import pinocchio as pin
from enum import Enum
from simulator.pybullet_wrapper import PybulletWrapper
import walking.talos_conf as conf
import pybullet as pb


class Side(Enum):
    """Side
    Describes which foot to use
    """
    LEFT=0
    RIGHT=1

def other_foot_id(id):
    if id == Side.LEFT:
        return Side.RIGHT
    else:
        return Side.LEFT
        
class FootStep:
    """FootStep
    Holds all information describing a single footstep
    """
    def __init__(self, pose, footprint, side=Side.LEFT):
        """inti FootStep

        Args:
            pose (pin.SE3): the pose of the footstep
            footprint (np.array): 3 by n matrix of foot vertices
            side (_type_, optional): Foot identifier. Defaults to Side.LEFT.
        """
        self.pose = pose
        self.footprint = footprint
        self.side = side
        self.line_ids = []
        
    def poseInWorld(self):
        return self.pose
        
    def plot(self, simulation):

        # Extract translation and rotation from pose
        pos = self.pose.translation
        rot = self.pose.rotation
        quat = pin.Quaternion(rot).coeffs()
        
        #>>>>TODO: plot in pybullet footprint, addGlobalDebugRectancle(...)
        world_points = pos.reshape(3,1) + rot @ self.footprint
        X = np.append(world_points[0,:], world_points[0,0])
        Y = np.append(world_points[1,:], world_points[1,0])
        Z = np.append(world_points[2,:], world_points[2,0])

        simulation.addGlobalDebugTrajectory(X,Y,Z, line_ids=self.line_ids, lifeTime=300)
        
        #>>>>TODO: display the side of the step, addUserDebugText(...)
        label_pos = pos + np.array([0.0, 0.0, 0.05])
        simulation.addUserDebugText(-1, -1, "L" if self.side == Side.LEFT else "R", label_pos, color=[0, 0, 0])

        #>>>>TODO: plot step target position addSphereMarker(...)
        simulation.addSphereMarker(pos, quat, radius=0.02, color=[1, 0, 0, 1])
        
        return None

class FootStepPlanner:
    """FootStepPlanner
    Creates footstep plans (list of right and left steps)
    """
    
    def __init__(self, conf):
        self.conf = conf
        self.steps = []
        
    def planLine(self, T_0_w, side, no_steps):
        """plan a sequence of steps in a strait line

        Args:
            T_0_w (pin.SE3): The inital starting position of the plan
            side (Side): The intial foot for starting the plan
            no_steps (_type_): The number of steps to take

        Returns:
            list: sequence of steps
        """
        
        # the displacement between steps in x and y direction
        dx = self.conf.step_size_x
        dy = 2*self.conf.step_size_y
        
        # the footprint of the robot
        lfxp, lfxn = self.conf.lfxp, self.conf.lfxn
        lfyp, lfyn = self.conf.lfyp, self.conf.lfyn

        footprint = np.array([
            [lfxp, lfxp, -lfxn, -lfxn],
            [lfyp, -lfyn, -lfyn, lfyp],
            [0.0, 0.0, 0.0, 0.0]
        ])
        
        #>>>>TODO: Plan a sequence of steps with T_0_w being the first step pose.
        #>>>>Note: Plan the second step parallel to the first step (robot starts standing on both feet)
        #>>>>Note: Plan the final step parallel to the last-1 step (robot stops standing on both feet)
        steps=[]

        # define first step
        steps.append(FootStep(T_0_w, footprint, side))

        # determine dy direction from first step
        dy_direction = 1
        if (side == Side.LEFT):
            dy_direction = -1

        # define start foot id
        foot_id = side

        for i in range(1,no_steps):

            # define foot id
            foot_id = other_foot_id(foot_id)

            y = 0
            if ((i%2) == 1):
                y = dy_direction*dy

            x = (i-1) * dx
            if (i == no_steps-1): 
                # define x-position of last step
                x = x - dx

            pose = pin.SE3(T_0_w.rotation, T_0_w.translation + T_0_w.rotation @ np.array([x, y, 0.0]))
            steps.append(FootStep(pose, footprint, foot_id))
                                
        self.steps = steps
        return steps

    
    def plot(self, simulation):
        for step in self.steps:
            step.plot(simulation)

def main():
    # setup the simulator
    simulator = PybulletWrapper(sim_rate=1000)

    #Needed for compatibility
    simulator.addLinkDebugFrame(-1,-1)

    # Setup pybullet camera
    pb.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=90,
        cameraPitch=-20,
        cameraTargetPosition=[0.0, 0.0, 0.8])
    
    #>>>>TODO: Generate a plan and plot it in pybullet.
    #>>>>TODO: Check that the plan looks as expected
    planer = FootStepPlanner(conf)
    planer.planLine(pin.SE3.Identity(), Side.RIGHT, 10)
    planer.plot(simulator)

    while True:
        # update the simulator and the robot
        simulator.step()
        simulator.debug()
                

if __name__=='__main__':
    """ Test footstep planner
    """
    main()
    


