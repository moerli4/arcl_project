import torch

class TorqueController:
    def __init__(
        self,
        robot,
    ):
        self.robot = robot
        self.K = 10
        self.D = 1
        
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

        f_ext_ee = state["f_ext_ee"]

        # compute dynamically consistent pseudo-inverse
        M_inv = torch.linalg.inv(M)
        J_cross = M_inv @ J.T @ torch.linalg.pinv(J @ M_inv @ J.T)

        # compute control torque
        tau = g

        return tau