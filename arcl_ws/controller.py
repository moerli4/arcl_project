import pinocchio as pin
import torch
import numpy as np

class TorqueController:
    def __init__(
        self,
        robot,
    ):
        self.robot = robot
        
    def compute_control_torque(self):
        """function to compute the control torques for the task

        Returns:
            torch tensor: tensor of desired control torques
        """
        state = self.robot.get_state()

        q = state["q"]
        q_dot = state["q_dot"]

        J = state["J"]
        M = state["M"]
        C = state["C"]
        g = state["g"]

        x = state["x"]
        x_dot = state["x_dot"]

        tau_measured = state["tau_measured"]

        # compute control torque
        tau = g

        return tau