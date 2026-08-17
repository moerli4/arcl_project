import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


class TorqueController:
    def __init__(
        self,
        robot,
        tasks,
    ):
        self.robot = robot
        self.tasks = tasks

    def compute_control_torque(self):
        J, f_des = None, None
        return J, f_des

    def plot_cartesian_pos_errors(self, dt):
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        for i, task in enumerate(self.tasks):
            if not hasattr(task, "error_hist"):
                continue

            errors = np.asarray(task.error_hist)
            time = np.arange(1, len(errors) + 1) * dt

            # x, y, z errors
            ax[0].plot(time, errors[:, 0], label=f"Task {i} - x")
            ax[0].plot(time, errors[:, 1], label=f"Task {i} - y", linestyle="--")
            ax[0].plot(time, errors[:, 2], label=f"Task {i} - z", linestyle=":")

            # Error norm
            error_norm = np.linalg.norm(errors, axis=1)
            ax[1].plot(time, error_norm, label=f"Task {i}")

        ax[0].axhline(0.0, linestyle="--", linewidth=0.8)
        ax[0].set_ylabel("Position error [m]")
        ax[0].set_title("Cartesian Position Errors")
        ax[0].grid(True)
        ax[0].legend()

        ax[1].set_xlabel("Time [s]")
        ax[1].set_ylabel(r"$\|e\|$ [m]")
        ax[1].set_title("Cartesian Position Error Norm")
        ax[1].grid(True)
        ax[1].legend()

        plt.tight_layout()
        plt.show()


class TorqueControllerQP(TorqueController):
    def __init__(
        self,
        robot,
        tasks,
    ):
        super().__init__(robot, tasks)

    def compute_control_torque(self):
        """function to compute the control torques for the task with the qp

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
            J_i, f_i = task.update(state)

            # initialize tau_i as the minimization variable
            tau_i = cp.Variable(n)

            # define quadratic minimization problem with cvxpy (Equation 18)
            objective = cp.Minimize(
                cp.sum_squares(J_i @ M_inv @ tau_i - J_i @ M_inv @ J_i.T @ f_i)
                + 1e-6 * cp.sum_squares(tau_i)
            )

            constraints = []

            # enforce robot torque limits with coriolis and gravity compensation (Equation 21)
            constraints = [
                tau_i >= self.robot.tau_min - h,
                tau_i <= self.robot.tau_max - h,
            ]

            # enforce all higher priority tasks as constraints (Equation 18)
            for j in range(i):
                J_j = J_task_history[j]
                tau_j = tau_opt_history[j]

                constraints.append(J_j @ M_inv @ tau_j == J_j @ M_inv @ tau_i)

            # solve for optimal tau
            problem = cp.Problem(objective, constraints)
            problem.solve()

            # save the task jacobian and optimal tau
            tau_opt_history[i] = tau_i.value
            J_task_history[i] = J_i

        # extract optimal tau
        tau_opt = tau_i.value

        # Calculate desired torque with coriolis and gravity compensation (Equation 20)
        tau_d = tau_opt + h

        return tau_d


class TorqueControllerTraditional(TorqueController):
    def __init__(
        self,
        robot,
        tasks,
    ):
        super().__init__(robot, tasks)

    def compute_control_torque(self):
        """function to compute the control torques for the task with traditional nullspace conntrol approach

        Returns:
            numpy array: desired control torques
        """

        # retrieve state
        state = self.robot.get_state()

        M = state["M"]
        h = state["h"]

        M_inv = np.linalg.inv(M)
        n = M.shape[0]

        # define tau with gravity+coriolis compensation
        tau_des = np.zeros(n)

        # nullspace projector N
        N = np.eye(n)

        for task in self.tasks:

            # get task Jacobian and desired force
            J, f = task.update(state)
            tau_task = J.T @ f

            # project task into nullspace
            J_proj = J @ N

            # calculate dynamically consistent pseudoinverse
            J_bar = M_inv @ J_proj.T @ np.linalg.pinv(J_proj @ M_inv @ J_proj.T)

            # add current task torque
            tau_des += N.T @ tau_task

            # update nullspace projector
            N = N @ (np.eye(n) - J_bar @ J_proj)

        # compensate nonlinear terms
        tau_des += h

        return tau_des
