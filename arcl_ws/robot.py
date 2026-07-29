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
        self.end_effector = self.genesis_robot_model.get_link(name="link7")
        self.end_effector_idx = self.end_effector.idx_local

        # torque limits
        self.tau_max = self.pin_robot_model.effortLimit
        self.tau_min = -self.tau_max

    def get_state(self):
        """Read the current robot state."""

        # read state from genesis robot model
        q = self.genesis_robot_model.get_dofs_position(self.robot_dofs_idx).cpu().numpy()
        q_dot = self.genesis_robot_model.get_dofs_velocity(self.robot_dofs_idx).cpu().numpy()
        J = self.genesis_robot_model.get_jacobian(link=self.end_effector).cpu().numpy()
        x = self.end_effector.get_pos().cpu().numpy()
        x_dot = self.end_effector.get_vel().cpu().numpy()
        f_ext_ee = self.genesis_robot_model.get_links_net_contact_force()[self.end_effector_idx].cpu().numpy()

        # calculate dynamics from pinocchio robot model

        M = pin.crba(
            self.pin_robot_model,
            self.pin_data,
            q,
        )
        h = pin.nonLinearEffects(
            self.pin_robot_model,
            self.pin_data,
            q,
            q_dot,
        )
        g = pin.computeGeneralizedGravity(
            self.pin_robot_model,
            self.pin_data,
            q,
        )
        C = h - g

        return {
            "q": q,
            "q_dot": q_dot,
            "J": J,
            "x": x,
            "x_dot": x_dot,
            "f_ext_ee": f_ext_ee,
            "M": M,
            "C": C,
            "g": g,
        }

    def command_torque(self, tau_cmd):
        """Send torque command to the simulated robot."""
        tau_cmd = torch.from_numpy(tau_cmd).float()

        self.genesis_robot_model.control_dofs_force(
                tau_cmd,
                self.robot_dofs_idx,
            )