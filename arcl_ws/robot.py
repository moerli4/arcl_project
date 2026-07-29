import pinocchio as pin
import torch
import numpy as np
import genesis as gs

class Robot():
    def __init__(self, scene, robot_xml_path):

        # add robot to the scene
        self.genesis_robot_model = scene.add_entity(
            gs.morphs.MJCF(file=robot_xml_path,
            pos   = (0.0, 0.0, 0.0),
            euler = (0, 0, 0),
            ),
        )
        jnt_names = ['joint1','joint2','joint3','joint4','joint5','joint6','joint7']
        self.robot_dofs_idx = [self.genesis_robot_model.get_joint(name).dof_idx_local for name in jnt_names]

        # create pinocchio robot model
        self.pin_robot_model = pin.buildModelsFromMJCF(filename=robot_xml_path)[0]
        self.pin_data = self.pin_robot_model.createData()

        # define end effector
        end_effector_link = "link7"
        self.end_effector = self.genesis_robot_model.get_link(end_effector_link)

    def get_state(self):
        """Read the current robot state."""

        # read state from genesis robot model
        q = self.genesis_robot_model.get_dofs_position(self.robot_dofs_idx)
        q_dot = self.genesis_robot_model.get_dofs_velocity(self.robot_dofs_idx)
        J = self.genesis_robot_model.get_jacobian(link=self.end_effector)
        x = self.end_effector.get_pos()
        x_dot = self.end_effector.get_vel()
        tau_measured = self.genesis_robot_model.get_dofs_force()

        # calculate dynamics from pinocchio robot model
        q_np = q.detach().cpu().numpy().astype(np.float64).reshape(-1)
        q_dot_np = q_dot.detach().cpu().numpy().astype(np.float64).reshape(-1)
        M = pin.crba(
            self.pin_robot_model,
            self.pin_data,
            q_np,
        )
        h = pin.nonLinearEffects(
            self.pin_robot_model,
            self.pin_data,
            q_np,
            q_dot_np,
        )
        g = pin.computeGeneralizedGravity(
            self.pin_robot_model,
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
            "tau_measured": tau_measured,
            "M": torch.from_numpy(M).to(device=q.device, dtype=q.dtype),
            "C": torch.from_numpy(C).to(device=q.device, dtype=q.dtype),
            "g": torch.from_numpy(g).to(device=q.device, dtype=q.dtype),
        }

    def command_torque(self, tau_cmd):
        """Send torque command to the simulated robot."""

        self.genesis_robot_model.control_dofs_force(
                tau_cmd,
                self.robot_dofs_idx,
            )