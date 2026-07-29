import numpy as np

class CylinderTask:
    """Task for moving the end effector onto a cylinder."""

    def __init__(self, radius, K, D):
        self.radius = radius
        self.K = K
        self.D = D

    def update(self, state):
        x = state["x"]
        x_dot = state["x_dot"]
        Jv = state["J"][:3, :]

        rho = np.sqrt(x[0]**2 + x[1]**2)

        radial_direction = np.array([
            x[0] / rho,
            x[1] / rho,
            0.0,
        ])

        self.J = radial_direction.reshape(1, 3) @ Jv

        rho_dot = radial_direction @ x_dot

        error = self.radius - rho
        error_dot = -rho_dot

        self.f_des = self.K * error + self.D * error_dot