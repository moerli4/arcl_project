import pinocchio as pin
import torch
import numpy as np

class TorqueController:
    def __init__(
        self,
        robot,
        robot_dofs_idx,
        pin_model,
        pin_data,
        end_effector_link_name,
    ):
        self.robot = robot
        self.robot_dofs_idx = robot_dofs_idx

        self.pin_model = pin_model
        self.pin_data = pin_data

        self.end_effector = robot.get_link(end_effector_link_name)

    def get_state(self):
        """Read the current robot state."""

        # read state from genesis robot model
        q = self.robot.get_dofs_position(self.robot_dofs_idx)
        q_dot = self.robot.get_dofs_velocity(self.robot_dofs_idx)
        J = self.robot.get_jacobian(link=self.end_effector)
        x = self.end_effector.get_pos()
        x_dot = self.end_effector.get_vel()

        # calculate dynamics from pinocchio robot model
        q_np = q.detach().cpu().numpy().astype(np.float64).reshape(-1)
        q_dot_np = q_dot.detach().cpu().numpy().astype(np.float64).reshape(-1)
        M = pin.crba(
            self.pin_model,
            self.pin_data,
            q_np,
        )
        h = pin.nonLinearEffects(
            self.pin_model,
            self.pin_data,
            q_np,
            q_dot_np,
        )
        g = pin.computeGeneralizedGravity(
            self.pin_model,
            self.pin_data,
            q_np,
        )
        C = h - g

        return {
            "q": q,
            "q_dot": q_dot,
            "J": J,
            "x": x,
            "x_dot": x_dot,
            "M": torch.from_numpy(M).to(device=q.device, dtype=q.dtype),
            "C": torch.from_numpy(C).to(device=q.device, dtype=q.dtype),
            "g": torch.from_numpy(g).to(device=q.device, dtype=q.dtype),
        }

    def compute_control_torque(self):
        """function to compute the control torques for the task

        Returns:
            torch tensor: tensor of desired control torques
        """
        state = self.get_state()

        q = state["q"]
        q_dot = state["q_dot"]

        J = state["J"]
        M = state["M"]
        C = state["C"]
        g = state["g"]

        x = state["x"]
        x_dot = state["x_dot"]

        # compute control torque
        tau = g

        return tau