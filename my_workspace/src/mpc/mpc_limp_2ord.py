def simulate(self, u, d=np.zeros(2)):
    """updates the LIP state x using based on command u
    
    Optionally: Takes a disturbance acceleration d to simulate effect
    of external pushes on the LIP.
    """

    #>>>>TODO: Compute x_dot and use euler integration to approximate
    # the state at t+dt
    #>>>>TODO: The disturbance is added in x_dot as self.D@d
    
    # Compute state derivative using continuous dynamics: x_dot = A*x + B*u + D*d
    x_dot = self.A @ self.x + self.B @ u + self.D @ d

    # Euler integration: x(t+dt) = x(t) + dt * x_dot(t)
    self.x = self.x + self.dt * x_dot

    return self.x