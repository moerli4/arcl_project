import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from time import perf_counter


class TorqueController:
    def __init__(
        self,
        robot,
        tasks,
    ):
        # robot stuff
        self.robot = robot
        self.tasks = tasks

        # plot stuff
        self.tau_cmd_hist = []

        # computation time parameters
        self._total_time = 0.0
        self._call_count = 0

    def compute_control_torque(self, *args, **kwargs):
        # wrapper for computation time benchmarking
        start = perf_counter()

        result = self._compute_control_torque(*args, **kwargs)

        elapsed = perf_counter() - start
        self._total_time += elapsed
        self._call_count += 1

        return result

    @property
    def average_time(self):
        # returns average computation time benchmark as a class attribute
        if self._call_count == 0:
            return 0.0
        return self._total_time / self._call_count

    def _compute_control_torque(self, *args, **kwargs):
        raise NotImplementedError

    def plot_cartesian_pos_errors(self, dt):
        # function to plot error and torque
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        for i, task in enumerate(self.tasks):
            if not hasattr(task, "error_hist"):
                continue

            errors = np.asarray(task.error_hist)
            time = np.arange(1, len(errors) + 1) * dt

            # x, y, z errors
            labels = ["x", "y", "z"]
            for j in range(3):
                if not np.allclose(errors[:, j], 0.0):
                    ax[0].plot(
                        time,
                        errors[:, j],
                        label=f"Task {i} - {labels[j]}",
                    )

        # Control Torque
        tau = np.array(self.tau_cmd_hist)
        for i in range(tau.shape[1]):
            ax[1].plot(
                time,
                tau[:, i],
            )
        ax[1].set_ylabel("Control Torque [Nm]")

        # labels etc
        ax[0].set_xlabel("Time [s]")
        ax[0].set_ylabel("Position error [m]")
        ax[0].set_title("Cartesian Position Errors")
        ax[0].grid(True)
        ax[0].legend(loc="upper right")

        ax[1].set_xlabel("Time [s]")
        ax[1].set_ylabel(r"$\tau$ [Nm]")
        ax[1].set_title("Control Torque")
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()


class TorqueControllerQP(TorqueController):
    def __init__(
        self,
        robot,
        tasks,
    ):
        super().__init__(robot, tasks)

    def _compute_control_torque(self):
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

            # enforce robot torque limits (Equation 19/21)
            constraints = [
                tau_i >= self.robot.tau_min,
                tau_i <= self.robot.tau_max,
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
                raise ValueError("Optimization problem is infeasible, try loosening torque limit constraints.")

            # save the task jacobian and optimal tau
            tau_opt_history[i] = tau_i.value
            J_task_history[i] = J_i

        # extract optimal tau
        tau_motion = tau_opt_history[-1]

        # compensate gravity and coriolis
        tau_cmd = tau_motion + h

        # save tau for plotting
        self.tau_cmd_hist.append(tau_cmd)

        return tau_cmd


class TorqueControllerTraditional(TorqueController):
    def __init__(
        self,
        robot,
        tasks,
    ):
        super().__init__(robot, tasks)

    def _compute_control_torque(self):
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

        # define tau
        tau_motion = np.zeros(n)

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
            tau_motion += N.T @ tau_task

            # update nullspace projector
            N = N @ (np.eye(n) - J_bar @ J_proj)

        # clip torques
        tau_motion = np.clip(tau_motion, self.robot.tau_min, self.robot.tau_max)

        # compensate gravity and coriolis
        tau_cmd = tau_motion + h

        # save tau for plotting
        self.tau_cmd_hist.append(tau_cmd)

        return tau_cmd
