import pinocchio as pin
import torch
import numpy as np
import genesis as gs


class Robot:
    def __init__(self, scene, robot_xml_path, gravity, ee_frame_name="attachment"):
        self.scene = scene

        # add robot to the scene
        self.genesis_robot_model = self.scene.add_entity(
            gs.morphs.MJCF(
                file=robot_xml_path,
                pos=(0.0, 0.0, 0.0),
                euler=(0, 0, 0),
            ),
            material=gs.materials.Rigid(
                gravity_compensation=0.0,
            ),
        )
        jnt_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.robot_dofs_idx = [
            self.genesis_robot_model.get_joint(name).dofs_idx_local[0]
            for name in jnt_names
        ]

        # create pinocchio robot model
        self.pin_robot_model = pin.buildModelsFromMJCF(filename=robot_xml_path)[0]
        self.pin_robot_model.gravity = pin.Motion(np.array(gravity), np.zeros(3))
        self.pin_data = self.pin_robot_model.createData()
        self.pin_ee_frame_id = self.pin_robot_model.getFrameId(ee_frame_name)
        self.pin_ee_joint_id = self.pin_robot_model.frames[
            self.pin_ee_frame_id
        ].parentJoint

        # define end effector
        self.genesis_end_effector = self.genesis_robot_model.get_link(
            name=ee_frame_name
        )
        self.genesis_end_effector_idx = self.genesis_end_effector.idx_local

    def retrieve_torque_limits(self, torque_limits=None):
        """Retrieve torque limits from genesis model, optionally provide artificial torque limits

        Args:
            torque_limits (list | None), optional: torque limits in form [None, 12, 100, None, ... , 10], overwrites default limits for valid entries

        Returns:
            tau_max (np.array): torque limits
        """

        # get torque limits from genesis model
        _, tau_max = self.genesis_robot_model.get_dofs_force_range()
        tau_max = tau_max.cpu().numpy()

        # override selected limits symmetrically
        if torque_limits is not None:
            for i, limit in enumerate(torque_limits):
                if limit is not None:
                    tau_max[i] = limit

        return tau_max

    def set_qpos(self, qpos):
        """Directly set a joint configuration."""

        self.genesis_robot_model.set_qpos(qpos, self.robot_dofs_idx)

    def set_pos(self, pos, quat=None):
        """Use inverse kinematics to directly set an ee position and quaternion."""

        qpos = self.genesis_robot_model.inverse_kinematics(
            link=self.genesis_end_effector, pos=pos, quat=quat
        )

        self.set_qpos(qpos)

    def get_state(self):
        """Read the current robot state."""

        # ----------------- read joint state from genesis robot model -----------------
        q = (
            self.genesis_robot_model.get_dofs_position(self.robot_dofs_idx)
            .cpu()
            .numpy()
        )
        q_dot = (
            self.genesis_robot_model.get_dofs_velocity(self.robot_dofs_idx)
            .cpu()
            .numpy()
        )

        # -------------- use pinocchio for calculations --------------
        # kinematics
        pin.forwardKinematics(
            self.pin_robot_model,
            self.pin_data,
            q,
            q_dot,
        )
        pin.updateFramePlacements(
            self.pin_robot_model,
            self.pin_data,
        )

        pin.computeJointKinematicHessians(
            self.pin_robot_model,
            self.pin_data,
            q,
        )

        # ee frame pose and velocity
        oM_ee = self.pin_data.oMf[self.pin_ee_frame_id]
        x = oM_ee.translation.copy()

        ee_velocity = pin.getFrameVelocity(
            self.pin_robot_model,
            self.pin_data,
            self.pin_ee_frame_id,
            pin.LOCAL_WORLD_ALIGNED,
        )
        x_dot = ee_velocity.linear.copy()

        ee_rotation = oM_ee.rotation.copy()
        quaternion = pin.Quaternion(ee_rotation).coeffs().copy()

        # jacobian
        J = pin.getFrameJacobian(
            self.pin_robot_model,
            self.pin_data,
            self.pin_ee_frame_id,
            pin.LOCAL_WORLD_ALIGNED,
        ).copy()

        # dynamics
        M = pin.crba(
            self.pin_robot_model,
            self.pin_data,
            q,
        )
        M = np.triu(M) + np.triu(M, 1).T

        h = pin.nonLinearEffects(
            self.pin_robot_model,
            self.pin_data,
            q,
            q_dot,
        )

        g = pin.computeGeneralizedGravity(self.pin_robot_model, self.pin_data, q)

        return {
            "q": q,
            "q_dot": q_dot,
            "J": J,
            "x": x,
            "ee_rotation": ee_rotation,
            "x_dot": x_dot,
            "quaternion": quaternion,
            "M": M,
            "h": h,
            "g": g,
        }

    def command_torque(self, tau_cmd):
        """Send torque command to the simulated robot."""
        tau_cmd = torch.from_numpy(tau_cmd).float()

        self.genesis_robot_model.control_dofs_force(
            tau_cmd,
            self.robot_dofs_idx,
        )
