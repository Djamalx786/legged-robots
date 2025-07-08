import numpy as np
import pinocchio as pin
import ndcurves as nc
import matplotlib.pyplot as plt

# import ndcurves, scipy, numpy, etc... to do your splines

class SwingFootTrajectory:
    """SwingFootTrajectory
    Interpolate Foot trajectory between SE3 T0 and T1
    """
    def __init__(self, T0, T1, duration, height=0.05):
        """initialize SwingFootTrajectory

        Args:
            T0 (pin.SE3): Inital foot pose
            T1 (pin.SE3): Final foot pose
            duration (float): step duration
            height (float, optional): setp height. Defaults to 0.05.
        """
        self._height = height
        self._t_elapsed = 0.0
        self._duration = duration
        self.reset(T0, T1)

    def reset(self, T0, T1):
        '''reset back to zero, update poses
        '''
        #>>>>TODO: plan the spline
        self.T0 = T0
        self.T1 = T1

        # Plan the translation spline
        p0 = T0.translation.copy()
        p1 = T1.translation.copy()

        # Add a middle point with the added height in Z
        pmid = 0.5 * (p0 + p1)
        pmid[2] += self._height

        # Build a spline for each axis with 3 key points: [0, mid, end]
        waypoints = np.vstack([p0, pmid, p1]).T 

        # create a curve with zero deriv constraints


    def isDone(self):
        return self._t_elapsed >= self._duration 
    
    def evaluate(self, t):
        """evaluate at time t
        """
        #>>>>TODO: evaluate the spline at time t, return pose, velocity, acceleration
        # ensure t is in valid range
        t = np.clip(t, 0.0, self._duration)

        # data: [pos(3), quat(4)]
        data = self.curve.evaluate_curve(t)
        pos = data[:3]
        quat = data[3:]

        # velocities/accelerations at t
        vel = self.curve.derivative(1, t)
        acc = self.curve.derivative(2, t)

        # return pose, linear velocity and acceleration
        pose = pin.SE3(pin.Quaternion(quat).matrix(), pos)
        vel_lin = vel[:3]
        acc_lin = acc[:3]

        return pose, vel_lin , acc_lin


if __name__=="__main__":
    T0 = pin.SE3(np.eye(3), np.array([0, 0, 0]))
    T1 = pin.SE3(np.eye(3), np.array([0.2, 0, 0]))

    #>>>>TODO: plot to make sure everything is correct
    traj = SwingFootTrajectory(T0, T1, duration=1.0, height=0.05)

    # get times for plot
    times = np.linspace(0, 1.0, 100)
    
    # get position, velocity, acceleration
    positions = []
    velocities = []
    accelerations = []

    for t in times:
        pose, vel, acc = traj.evaluate(t)
        positions.append(pose.translation)
        velocities.append(vel)
        accelerations.append(acc)

    positions = np.array(positions)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    labels = ['x', 'y', 'z']
    for i in range(3):
        axes[0].plot(times, positions[:, i], label=f'pos {labels[i]}')
        axes[1].plot(times, velocities[:, i], label=f'vel {labels[i]}')
        axes[2].plot(times, accelerations[:, i], label=f'acc {labels[i]}')

    axes[0].set_ylabel("Position (m)")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[2].set_ylabel("Acceleration (m/s²)")
    axes[2].set_xlabel("Time (s)")

    for ax in axes:
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


