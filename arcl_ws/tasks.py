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

        self.error_hist = []

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

        self.error_hist.append(error)

        return J, np.atleast_1d(f_des)


class CylinderTask:
    """Use virtual joint to constrain the end effector to a vertical cylinder of specified radius around a specified point in x-y-plane."""

    def __init__(self, robot, radius, center, scene, ee_frame_name="attachment"):
        self.K = 20.0
        self.D = 10.0

        self.model = robot.pin_robot_model.copy()

        assert len(center) == 2  # (x,y)
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
        z_dot_jacobian = ez @ (-pin.skew(z)) @ J_angular
        velocity_z = z_dot_jacobian @ v
        f_z = self.K * error_z - self.D * velocity_z

        # combine position and orientation
        J = np.vstack(
            [
                J_xy,
                z_dot_jacobian,
            ]
        )
        f_des = np.concatenate(
            [
                f_xy,
                np.atleast_1d(f_z),
            ]
        )

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

        grad_w = -(q - self.q_mid) / (self.model.nv * self.q_range**2)

        J_task = np.eye(self.model.nv)
        f_des = self.K * grad_w - self.D * q_dot

        return J_task, np.atleast_1d(f_des)


class MaximizeManipulabilityTask:
    """Maximize manipulability by maximizing yoshikawa measure."""

    def __init__(self, robot):
        self.model = robot.pin_robot_model
        self.data = robot.pin_data
        self.ee_id = robot.pin_ee_joint_id
        self.K = 1.0
        self.D = 0.2

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
        self,
        p0,
        p1,
        period,
        scene,
        dt,
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

        f_des = self.K * error + self.D * (target_velocity - velocity)

        return J, f_des


class HorizontalPlaneTask:
    """Constrain ee to a horizontal height and keep ee orthogonal ie pointing downwards"""

    def __init__(
        self,
        height,
    ):
        self.K = 20.0
        self.D = 10.0

        self.height = height

        self.error_hist = []

    def update(self, state):
        v = state["q_dot"]
        p = state["x"]
        z = state["ee_rotation"][:, 2]

        J6 = state["J"]

        Jz = -pin.skew(z) @ J6[3:, :]

        J = np.vstack(
            [
                J6[2, :],
                Jz[0, :],
                Jz[1, :],
            ]
        )

        # position error
        pos_error_z = self.height - p[2]

        # compose desired force from position and orientation error
        f_des = np.array(
            [
                self.K * pos_error_z - self.D * (J6[2, :] @ v),
                -self.K * z[0] - self.D * (Jz[0, :] @ v),
                -self.K * z[1] - self.D * (Jz[1, :] @ v),
            ]
        )

        # save stuff for plotting
        self.error_hist.append([0, 0, pos_error_z])

        return J, np.atleast_1d(f_des)


class PointSequenceTask:
    """move ee through trajectory defined by waypoint sequence"""

    def __init__(self, points, segment_time, scene, dt):
        self.K = 20.0
        self.D = 10.0

        self.points = np.asarray(points, dtype=float)
        self.segment_time = segment_time

        self.time = 0.0
        self.dt = dt

        for p in self.points:
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

        self.error_hist = []

    def update(self, state):
        self.time += self.dt

        p = state["x"]
        v = state["q_dot"]

        J = state["J"][:3, :]
        velocity = J @ v

        # retrieve current segment
        segment = min(
            int(self.time / self.segment_time),
            len(self.points) - 2,
        )

        # find target ee pos and vel
        s = (self.time - segment * self.segment_time) / self.segment_time
        s = min(s, 1.0)
        p0 = self.points[segment]
        p1 = self.points[segment + 1]
        target = p0 + s * (p1 - p0)
        target_velocity = (p1 - p0) / self.segment_time

        # hold final point
        if self.time >= (len(self.points) - 1) * self.segment_time:
            target = self.points[-1]
            target_velocity = np.zeros(3)

        # cartesian pd
        error = target - p
        f_des = self.K * error + self.D * (target_velocity - velocity)

        # save stuff for plotting
        self.error_hist.append(error)

        return J, f_des


class XPositionTask:
    """Control the end-effector x position."""

    def __init__(self, x_des):
        self.K = 20.0
        self.D = 10.0

        self.x_des = x_des
        self.error_hist = []

    def update(self, state):
        q_dot = state["q_dot"]
        p = state["x"]
        J6 = state["J"]

        # x translational Jacobian
        J = J6[0:1, :]

        error = self.x_des - p[0]
        velocity = J6[0, :] @ q_dot

        f_des = self.K * error - self.D * velocity

        self.error_hist.append([error, 0.0, 0.0])

        return J, np.atleast_1d(f_des)


class YPositionTask:
    """Control the end-effector y position."""

    def __init__(self, y_des):
        self.K = 20.0
        self.D = 10.0

        self.y_des = y_des
        self.error_hist = []

    def update(self, state):
        q_dot = state["q_dot"]
        p = state["x"]
        J6 = state["J"]

        # y translational Jacobian
        J = J6[1:2, :]

        error = self.y_des - p[1]
        velocity = J6[1, :] @ q_dot

        f_des = self.K * error - self.D * velocity

        self.error_hist.append([0.0, error, 0.0])

        return J, np.atleast_1d(f_des)


class ZSinusoidalTask:
    """Track a sinusoidal trajectory along the end-effector z axis."""

    def __init__(
        self,
        z_center,
        amplitude,
        frequency,
        dt,
    ):
        self.K = 20.0
        self.D = 10.0

        self.z_center = z_center
        self.amplitude = amplitude
        self.frequency = frequency
        self.dt = dt

        self.time = 0.0

        self.error_hist = []
        self.desired_position_hist = []

    def update(self, state):
        q_dot = state["q_dot"]
        p = state["x"]
        J6 = state["J"]

        # Angular frequency
        omega = 2.0 * np.pi * self.frequency

        # Desired z position
        z_des = self.z_center + self.amplitude * np.sin(omega * self.time)

        # Desired z velocity
        z_dot_des = self.amplitude * omega * np.cos(omega * self.time)

        # z translational Jacobian
        J = J6[2:3, :]

        z_dot = J6[2, :] @ q_dot

        error = z_des - p[2]

        f_des = self.K * error + self.D * (z_dot_des - z_dot)

        self.error_hist.append([0.0, 0.0, error])
        self.desired_position_hist.append(z_des)

        self.time += self.dt

        return J, np.atleast_1d(f_des)
