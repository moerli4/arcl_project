import numpy as np
import cvxpy as cp


class TorqueController:
    def __init__(self):
        pass

    def compute_control_torque(self, *args, **kwargs):
        raise NotImplementedError


class TorqueControllerQP(TorqueController):
    def __init__(self, epsilon=1e-6):
        super().__init__()

        # regularisation factor epsilon
        self.epsilon = epsilon

    def compute_control_torque(self, state, tasks, torque_limits):
        """Function to compute the control torques for the task with the QP, as seen in the paper.

        Receives:
            state (dict): state from robot
            tasks (list): list of control objectives in order of priority
            torque_limits (list): maximum allowed joint torque

        Returns:
            numpy array: desired control torques
        """

        # retrieve dynamics
        M = state["M"]
        h = state["h"]

        M_inv = np.linalg.inv(M)
        n = M.shape[0]

        # save task jacobian and optimal torque for each task
        J_task_history = [None] * len(tasks)
        tau_opt_history = [None] * len(tasks)

        # iterate over all tasks in order
        for i, task in enumerate(tasks):

            # update task and retrieve task jacobian and desired force f_i
            J_i, f_i = task.update(state)

            # initialize tau_i as the minimization variable
            tau_i = cp.Variable(n)

            # define quadratic minimization problem with cvxpy (Equation 18)
            objective = cp.Minimize(
                cp.sum_squares(J_i @ M_inv @ tau_i - J_i @ M_inv @ J_i.T @ f_i)
                + self.epsilon * cp.sum_squares(tau_i)
            )

            constraints = []

            # enforce robot torque limits (Equation 19/21)
            constraints = [
                tau_i >= -torque_limits - h,
                tau_i <= torque_limits - h,
            ]

            # enforce all higher priority tasks as constraints (Equation 18)
            for j in range(i):
                J_j = J_task_history[j]
                tau_j = tau_opt_history[j]

                constraints.append(J_j @ M_inv @ tau_j == J_j @ M_inv @ tau_i)

            # solve for optimal tau
            problem = cp.Problem(objective, constraints)
            problem.solve(verbose=False)
            if problem.status == cp.INFEASIBLE:
                raise ValueError(
                    "Optimization problem is infeasible, try loosening torque limit constraints."
                )

            # save the task jacobian and optimal tau
            tau_opt_history[i] = tau_i.value
            J_task_history[i] = J_i

        # extract optimal tau
        tau_motion = tau_opt_history[-1]

        # compensate gravity and coriolis
        tau_cmd = tau_motion + h

        return tau_cmd


class TorqueControllerTraditional(TorqueController):
    def __init__(self):
        super().__init__()

    def compute_control_torque(self, state, tasks, torque_limits):
        """function to compute the control torques for the task with traditional nullspace conntrol approach

        Receives:
            state (dict): state from robot
            tasks (list): list of control objectives in order of priority
            torque_limits (list): maximum allowed joint torque

        Returns:
            numpy array: desired control torques
        """

        # retrieve dynamics
        M = state["M"]
        h = state["h"]

        M_inv = np.linalg.inv(M)
        n = M.shape[0]

        # define tau
        tau_motion = np.zeros(n)

        # nullspace projector N
        N = np.eye(n)

        for task in tasks:

            # get task Jacobian and desired force
            J, f = task.update(state)
            tau_task = J.T @ f

            # project task into nullspace
            J_proj = J @ N

            # calculate dynamically consistent pseudoinverse
            J_bar = M_inv @ J_proj.T @ np.linalg.pinv(J_proj @ M_inv @ J_proj.T)

            # add current task torque
            tau_motion += N.T @ tau_task

            # update nullspace projector
            N = N @ (np.eye(n) - J_bar @ J_proj)

        # compensate gravity and coriolis
        tau_cmd = tau_motion + h

        # clip torques
        tau_cmd = np.clip(tau_cmd, -torque_limits, torque_limits)

        return tau_cmd
