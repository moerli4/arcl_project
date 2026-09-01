import numpy as np

import pinocchio as pin
import genesis as gs


class Task:
    def __init__(self, K, D):
        self.error_hist = []
        self.K = K
        self.D = D

    def reset(self):
        self.error_hist = []


class XPositionTask(Task):
    """Move end effector to a desired x."""

    def __init__(self, x_des, K, D):
        super().__init__(K=K, D=D)

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


class YPositionTask(Task):
    """Move end effector to a desired y."""

    def __init__(self, y_des, K, D):
        super().__init__(K=K, D=D)

        self.y_des = y_des
        self.error_hist = []

    def update(self, state):
        q_dot = state["q_dot"]
        p = state["x"]
        J6 = state["J"]

        # task jacobian
        J = J6[1:2, :]

        error = self.y_des - p[1]
        velocity = J6[1, :] @ q_dot

        f_des = self.K * error - self.D * velocity

        self.error_hist.append([0.0, error, 0.0])

        return J, np.atleast_1d(f_des)


class ZSinusoidalTask(Task):
    """Track a sinusoidal trajectory along the endeffectors z axis."""

    def __init__(self, z_center, amplitude, frequency, dt, K, D):
        super().__init__(K=K, D=D)

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

        # angular frequency
        omega = 2.0 * np.pi * self.frequency

        # desired z
        z_des = self.z_center + self.amplitude * np.sin(omega * self.time)

        # desired z velocity
        z_dot_des = self.amplitude * omega * np.cos(omega * self.time)

        # task jacobian
        J = J6[2:3, :]

        z_dot = J6[2, :] @ q_dot

        error = z_des - p[2]

        f_des = self.K * error + self.D * (z_dot_des - z_dot)

        self.error_hist.append([0.0, 0.0, error])
        self.desired_position_hist.append(z_des)

        self.time += self.dt

        return J, np.atleast_1d(f_des)


class SphereTask(Task):
    """Use virtual joint to constrain the end effector to a sphere of specified radius around a specified point."""

    def __init__(
        self,
        robot,
        radius,
        center,
        scene,
        K,
        D,
        ee_frame_name="attachment",
    ):
        super().__init__(K=K, D=D)

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

        self.error_hist.append(error)

        return J, np.atleast_1d(f_des)


class MaximizeManipulabilityTask(Task):
    """Maximize manipulability by maximizing yoshikawa measure."""

    def __init__(self, robot):
        super().__init__(K=1, D=0.2)

        self.model = robot.pin_robot_model
        self.data = robot.pin_data
        self.ee_id = robot.pin_ee_joint_id

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
