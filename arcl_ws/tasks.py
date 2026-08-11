import numpy as np
import pinocchio as pin
import genesis as gs

class SphereTask:
    """Use virtual joint to constrain the end effector to a sphere of specified radius around a specified point."""

    def __init__(self, robot, radius, center, scene, ee_frame_name="attachment"):
        self.K = 20.0
        self.D = 10.0

        self.model = robot.pin_robot_model.copy()
        self.center = center

        # create virtual joint link
        ee_frame_id = self.model.getFrameId(ee_frame_name)
        ee_frame = self.model.frames[ee_frame_id]
        self.virtual_frame_id = self.model.addFrame(
            pin.Frame(
                "virtual_frame",
                ee_frame.parentJoint,
                ee_frame_id,
                ee_frame.placement * pin.SE3(np.eye(3), np.array([0.0, 0.0, radius])),
                pin.FrameType.OP_FRAME,
            )
        )
        self.model_data = self.model.createData()

        # add visualization     
        scene.add_entity(
            gs.morphs.Sphere(
                pos=center,
                radius=radius,
                fixed=True,
                collision=False,
                visualization=True,
            ),
            surface=gs.surfaces.Default(
                color=(1.0, 0.0, 0.0, 0.25),
            ),
        )

    def update(self, state):

        q = state["q"]
        v = state["q_dot"]
        
        pin.forwardKinematics(self.model, self.model_data, q, v)
        pin.updateFramePlacements(self.model, self.model_data)
        pin.computeJointJacobians(self.model, self.model_data, q)

        position = self.model_data.oMf[self.virtual_frame_id].translation

        J6 = pin.getFrameJacobian(
            self.model,
            self.model_data,
            self.virtual_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        J = J6[:3, :]
        velocity = J @ v

        error = self.center - position
        f_des = self.K * error - self.D * velocity

        return J, np.atleast_1d(f_des)


class CylinderTask:
    """Use virtual joint to constrain the end effector to a vertical cylinder of specified radius around a specified point in x-y-plane."""

    def __init__(self, robot, radius, center, scene, ee_frame_name="attachment"):
        self.K = 20.0
        self.D = 10.0

        self.model = robot.pin_robot_model.copy()

        assert len(center)==2 # (x,y)
        self.center = center

        # create virtual joint link
        ee_frame_id = self.model.getFrameId(ee_frame_name)
        ee_frame = self.model.frames[ee_frame_id]
        self.virtual_frame_id = self.model.addFrame(
            pin.Frame(
                "virtual_frame",
                ee_frame.parentJoint,
                ee_frame_id,
                ee_frame.placement * pin.SE3(np.eye(3), np.array([0.0, 0.0, radius])),
                pin.FrameType.OP_FRAME,
            )
        )
        self.model_data = self.model.createData()

        # add visualization     
        scene.add_entity(
            gs.morphs.Cylinder(
                pos=np.array([center[0], center[1], 0]),
                radius=radius,
                height=5,
                fixed=True,
                collision=False,
                visualization=True,
            ),
            surface=gs.surfaces.Default(
                color=(1.0, 0.0, 0.0, 0.25),
            ),
        )

    def update(self, state):

        q = state["q"]
        v = state["q_dot"]
        
        pin.forwardKinematics(self.model, self.model_data, q, v)
        pin.updateFramePlacements(self.model, self.model_data)
        pin.computeJointJacobians(self.model, self.model_data, q)

        position = self.model_data.oMf[self.virtual_frame_id].translation
        R = self.model_data.oMf[self.virtual_frame_id].rotation

        J6 = pin.getFrameJacobian(
            self.model,
            self.model_data,
            self.virtual_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        # position: on the cylinder
        J_xy = J6[:2, :]
        velocity_xy = J_xy @ v
        error_xy = self.center - position[:2]
        f_xy = self.K * error_xy - self.D * velocity_xy

        # orientation: orthogonal to cylinder (horizontal z)
        z = R[:, 2]
        error_z = -z[2]
        J_angular = J6[3:, :]
        ez = np.array([0.0, 0.0, 1.0])
        z_dot_jacobian = (
            ez @ (-pin.skew(z)) @ J_angular
        )
        velocity_z = z_dot_jacobian @ v
        f_z = self.K * error_z - self.D * velocity_z

        # combine position and orientation
        J = np.vstack([
            J_xy,
            z_dot_jacobian,
        ])
        f_des = np.concatenate([
            f_xy,
            np.atleast_1d(f_z),
        ])

        return J, np.atleast_1d(f_des)


class AvoidJointLimitsTask:
    """Keep joints near the middle of their joint limits."""

    def __init__(self, robot):
        self.model = robot.pin_robot_model

        self.K = 5
        self.D = 5

        self.q_min = self.model.lowerPositionLimit
        self.q_max = self.model.upperPositionLimit
        self.q_mid = 0.5 * (self.q_min + self.q_max)
        self.q_range = self.q_max - self.q_min

    def update(self, state):
        q = state["q"]
        q_dot = state["q_dot"]

        grad_w = -(q - self.q_mid) / (
            self.model.nv * self.q_range**2
        )

        J_task = np.eye(self.model.nv)
        f_des = self.K * grad_w - self.D * q_dot

        return J_task, np.atleast_1d(f_des)


class MaximizeManipulabilityTask:
    """Maximize manipulability by maximizing yoshikawa measure."""
    def __init__(self, robot):
        self.model = robot.pin_robot_model
        self.data = robot.pin_data
        self.ee_id = robot.pin_ee_joint_id
        self.K = 2.0
        self.D = .5

    def update(self, state):
        # manipulator J and H
        J = pin.getJointJacobian(
            self.model,
            self.data,
            self.ee_id,
            pin.LOCAL_WORLD_ALIGNED,
        )[:3, :]

        H = pin.getJointKinematicHessian(
            self.model,
            self.data,
            self.ee_id,
            pin.LOCAL_WORLD_ALIGNED,
        )[:3, :, :]

        q_dot = state["q_dot"]

        w = np.sqrt(max(0.0, np.linalg.det(J @ J.T)))
        J_pinv = np.linalg.pinv(J)

        grad_w = w * np.einsum(
            "ab,bia->i",
            J_pinv,
            H,
        )

        J_task = np.eye(self.model.nv)
        f_des = self.K * grad_w - self.D * q_dot

        return J_task, np.atleast_1d(f_des)


class PointToPointTask:
    """Move the ee periodically between two cartesian points."""

    def __init__(
        self, p0, p1, period, scene, dt,
    ):
        self.K = 20.0
        self.D = 10.0

        self.p0 = np.asarray(p0, dtype=float)
        self.p1 = np.asarray(p1, dtype=float)
        self.period = period
        self.time = 0.0
        self.dt = dt

        for p in [self.p0, self.p1]:
            scene.add_entity(
                gs.morphs.Sphere(
                    pos=p,
                    radius=0.03,
                    fixed=True,
                    collision=False,
                    visualization=True,
                ),
                surface=gs.surfaces.Default(
                    color=(0.0, 1.0, 0.0, 0.5),
                ),
            )

    def update(self, state):
        v = state["q_dot"]

        dt = self.dt
        self.time += dt

        p = state["x"]
        J6 = state["J"]

        J = J6[:3, :]
        velocity = J @ v

        # oscillate between p0 and p1
        phase = 2.0 * np.pi * self.time / self.period

        alpha = 0.5 * (1.0 - np.cos(phase))
        alpha_dot = 0.5 * (2.0 * np.pi / self.period) * np.sin(phase)

        target = self.p0 + alpha * (self.p1 - self.p0)
        target_velocity = alpha_dot * (self.p1 - self.p0)

        error = target - p

        f_des = (
            self.K * error
            + self.D * (target_velocity - velocity)
        )

        return J, f_des


class HorizontalPlaneTask:
    """Constrain ee to a horizontal height and keep ee orthogonal ie pointing downwards
    """

    def __init__(
        self,
        height,
    ):
        self.K = 20.0
        self.D = 10.0

        self.height = height

    def update(self, state):
        v = state["q_dot"]
        p = state["x"]
        z = state["ee_rotation"][:, 2]

        J6 = state["J"]

        Jz = -pin.skew(z) @ J6[3:, :]

        J = np.vstack([
            J6[2, :], 
            Jz[0, :],
            Jz[1, :], 
        ])

        f_des = np.array([
            self.K * (self.height - p[2]) - self.D * (J6[2, :] @ v),
            -self.K * z[0] - self.D * (Jz[0, :] @ v),
            -self.K * z[1] - self.D * (Jz[1, :] @ v),
        ])

        return J, np.atleast_1d(f_des)