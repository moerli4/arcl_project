import torch


class TorqueController:
    def __init__(
        self,
        robot,
        robot_dofs_idx,
        end_effector_link_name="hand",
    ):
        self.robot = robot
        self.robot_dofs_idx = robot_dofs_idx
        self.end_effector = robot.get_link(end_effector_link_name)

    def get_state(self):
        dofs = self.robot_dofs_idx

        # Joint state
        q = self.robot.get_dofs_position(dofs)
        q_dot = self.robot.get_dofs_velocity(dofs)

        # Dynamics
        M_full = self.robot.get_mass_mat()
        M = M_full[dofs][:, dofs]

        # Jacobian
        J_full = self.robot.get_jacobian(link=self.end_effector)
        J = J_full[:, dofs]

        
        # Cartesian state
        x = self.end_effector.get_pos()
        quat = self.end_effector.get_quat()

        x_dot = self.end_effector.get_vel()
        omega = self.end_effector.get_ang()

        # Diagnostics
        tau_cmd = self.robot.get_dofs_control_force(dofs)
        tau_measured = self.robot.get_dofs_force(dofs)

        return {
            "q": q,
            "q_dot": q_dot,
            "M": M,
            "J": J,
            "J_linear": J[:3],
            "J_angular": J[3:],
            "x": x,
            "quat": quat,
            "x_dot": x_dot,
            "omega": omega,
            "tau_cmd": tau_cmd,
            "tau_measured": tau_measured,
        }

    def compute_control_torque(self):
        state = self.get_state()

        q = state["q"]
        q_dot = state["q_dot"]
        M = state["M"]
        J = state["J_linear"]

        tau = torch.zeros(
            len(self.robot_dofs_idx),
            device=q.device,
            dtype=q.dtype,
        )

        return tau