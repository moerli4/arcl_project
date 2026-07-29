import numpy as np


class DemoTask:
    """Simple Cartesian point-to-point task."""

    def __init__(self):
        self.K = 10.0
        self.D = 2.0

        # target position in world coordinates
        self.target = np.array([
            0.5,
            0.5,
            0.1,
        ])

    def update(self, state):

        x = state["x"]

        Jv = state["J"][:3, :]
        x_dot = Jv @ state["q_dot"]

        error = self.target - x
        error_dot = -x_dot

        self.J = Jv
        self.f_des = (
            self.K * error
            + self.D * error_dot
        )