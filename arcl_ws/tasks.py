import numpy as np
import pinocchio as pin


class DemoTask:
    """Simple Cartesian point-to-point task."""

    def __init__(self):
        self.K = 10.0
        self.D = 2.0

        # target position in world coordinates
        self.target = np.array(
            [
                0.5,
                0.5,
                0.1,
            ]
        )

    def update(self, state):

        x = state["x"]

        Jv = state["J"][:3, :]
        x_dot = Jv @ state["q_dot"]

        error = self.target - x
        error_dot = -x_dot

        f_des = self.K * error + self.D * error_dot

        return Jv, np.atleast_1d(f_des)


class SphereTask:
    """Use virtual joint to constrain the end effector to a sphere of specified radius around a specified point."""

    def __init__(self, robot, radius, center, ee_frame_name="attachment"):
        self.K = 20.0
        self.D = 10.0

        self.model = robot.pin_robot_model.copy()
        self.center = center

        # create virtual joint link at sphere center
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

# class AvoidJointLimitsTask:
#     """Keep joints near the middle of their joint limits."""

#     def __init__(self, robot):
#         self.model = robot.pin_robot_model

#         self.K = 5
#         self.D = 5

#         self.q_min = self.model.lowerPositionLimit
#         self.q_max = self.model.upperPositionLimit
#         self.q_mid = 0.5 * (self.q_min + self.q_max)
#         self.q_range = self.q_max - self.q_min

#     def update(self, state):
#         q = state["q"]
#         q_dot = state["q_dot"]

#         grad_w = -(q - self.q_mid) / (
#             self.model.nv * self.q_range**2
#         )

#         J = np.eye(self.model.nv)
#         f_des = self.K * grad_w - self.D * q_dot

#         return J, np.atleast_1d(f_des)

# class MaximizeManipulabilityTask:
#     """Maximize manipulability by maximizing yoshikawa measure."""
#     def __init__(self, robot):
#         self.model = robot.pin_robot_model
#         self.K = 10.0
#         self.D = 2.0

#     def update(self, state):
#         J = state["J_manip"][:3, :]
#         H = state["H_manip"][:3, :, :]
#         q_dot = state["q_dot"]

#         w = np.sqrt(max(0.0, np.linalg.det(J @ J.T)))
#         J_pinv = np.linalg.pinv(J)

#         grad_w = w * np.einsum(
#             "ab,bai->i",
#             J_pinv,
#             H,
#         )

#         J_task = np.eye(self.model.nv)
#         f_des = self.K * grad_w - self.D * q_dot

#         return J_task, np.atleast_1d(f_des)