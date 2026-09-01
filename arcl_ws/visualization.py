import numpy as np
import matplotlib.pyplot as plt


def plot_cartesian_pos_errors(tasks, tau, torque_limits, dt=0.01):
    """Function to plot cartesian position error and control torque over time."""

    _, ax = plt.subplots(3, 1, figsize=(15, 7), sharex=True)

    # ------- Cartesian position error -------
    for i, task in enumerate(tasks):
        if len(task.error_hist) == 0:
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
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Position error [m]")
    ax[0].set_title("Cartesian Position Errors")
    ax[0].grid(True)
    ax[0].legend(loc="upper right")

    # ------- Control Torque -------
    for i in range(tau.shape[1]):
        ax[1].plot(
            time,
            tau[:, i],
            label=f"Joint {i + 1}",
        )
    ax[1].set_ylabel(r"$\tau$")
    ax[1].set_title("Control Torque")
    ax[1].legend(loc="upper right")

    # ------- Control Torque normalized to torque limits -------
    tau_normalized = tau / torque_limits[None, :]

    for i in range(tau_normalized.shape[1]):
        ax[2].plot(
            time,
            tau_normalized[:, i],
            label=f"Joint {i + 1}",
        )
    ax[2].axhline(1.0, linestyle="--")
    ax[2].axhline(-1.0, linestyle="--")
    ax[2].set_ylabel(r"$\tau / \tau_{\max}$")
    ax[2].set_title("Normalized Control Torque")
    ax[2].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

def print_metrics(
    tau_cmd_hist,
    tasks,
    computation_times,
    dt=0.01,
    transient_time=5.0,
):
    tau_cmd_hist = np.asarray(tau_cmd_hist)
    computation_times = np.asarray(computation_times)

    start_idx = int(transient_time / dt)

    # computation
    comp_time_ms = np.mean(computation_times) * 1e3

    # torque
    tau_rms = np.sqrt(np.mean(tau_cmd_hist**2))

    print("-----------------------------")
    print(f"Computation time : {comp_time_ms:8.3f} ms")
    print(f"RMS torque       : {tau_rms:8.3f} Nm")

    # tracking
    for i, task in enumerate(tasks):
        error_hist = np.asarray(task.error_hist)[start_idx:]
        error_norm = np.linalg.norm(error_hist, axis=1)

        rmse = np.sqrt(np.mean(error_norm**2))
        max_error = np.max(np.abs(error_hist))

        print(f"Task {i} RMSE     : {rmse:8.4f} m")
        print(f"Task {i} Max Error     : {max_error:8.4f} m")
