import numpy as np
import cvxpy as cp

from tasks import DemoTask


class TorqueController:
    def __init__(
        self,
        robot,
    ):
        self.robot = robot

        # define tasks in order of priority
        self.tasks = [
            DemoTask()
        ]

    def compute_control_torque(self):
        """function to compute the control torques for the task

        Returns:
            numpy array: desired control torques
        """

        # read robot state
        state = self.robot.get_state()

        M = state["M"]
        h = state["h"]

        M_inv = np.linalg.inv(M)
        n = M.shape[0]

        # save task jacobian and optimal torque for each task
        J_task_history = [None] * len(self.tasks)
        tau_opt_history = [None] * len(self.tasks)

        # iterate over all tasks in order
        for i, task in enumerate(self.tasks):

            # update task and retrieve task jacobian and desired force f_i
            task.update(state)
            J_i = task.J
            f_i = np.atleast_1d(task.f_des)

            # initialize tau_i as the minimization variable
            tau_i = cp.Variable(n)

            # define quadratic minimization problem with cvxpy (Equation 18)
            objective = cp.Minimize(
                cp.sum_squares(J_i @ M_inv @ tau_i - J_i @ M_inv @ J_i.T @ f_i) + 1e-6 * cp.sum_squares(tau_i)
            )

            constraints = []

            # enforce robot torque limits with coriolis and gravity compensation (Equation 21)
            constraints = [
                tau_i >= self.robot.tau_min - (h),
                tau_i <= self.robot.tau_max - (h),
            ]
            
            # enforce all higher priority tasks as constraints (Equation 18)
            for j in range(i):
                J_j = J_task_history[j]
                tau_j = tau_opt_history[j]

                constraints.append(
                    J_j @ M_inv @ tau_j
                    ==
                    J_j @ M_inv @ tau_i
                )

            # solve for optimal tau
            problem = cp.Problem(objective, constraints)
            problem.solve()

            # save the task jacobian and optimal tau
            tau_opt_history[i] = tau_i.value
            J_task_history[i] = task.J

        # extract optimal tau
        tau_opt = tau_opt_history[-1] 

        # Calculate desired torque with coriolis and gravity compensation (Equation 20)
        tau_d = tau_opt + h

        return tau_d