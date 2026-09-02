import numpy as np
import matplotlib.pyplot as plt


def plot_results(
    ax,
    controller_results: dict,
    dt: float,
    show_normalized_torque: bool,
    controller_type: str,
):
    task_error_hists = controller_results["task_error_hists"]
    tau_cmd_hist = controller_results["tau_cmd_hist"]
    torque_limits = controller_results["torque_limits"]

    # ------- Cartesian position error -------

    for i, errors in enumerate(task_error_hists):
        if errors.size == 0:
            continue

        time = np.arange(1, len(errors) + 1) * dt

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
    ax[0].set_title(f"{controller_type}\nCartesian Position Errors")
    ax[0].grid(True)
    ax[0].legend(loc="upper right")

    # ------- Control Torque -------

    torque_time = np.arange(1, len(tau_cmd_hist) + 1) * dt

    for i in range(tau_cmd_hist.shape[1]):
        ax[1].plot(
            torque_time,
            tau_cmd_hist[:, i],
        )

    ax[1].set_ylabel(r"$\tau$ [Nm]")
    ax[1].set_title("Control Torque")
    ax[1].grid(True)

    # ------- Normalized Control Torque -------

    if show_normalized_torque:
        tau_normalized = tau_cmd_hist / torque_limits[None, :]

        for i in range(tau_normalized.shape[1]):
            ax[2].plot(
                torque_time,
                tau_normalized[:, i],
            )

        ax[2].axhline(1.0, linestyle="--")
        ax[2].axhline(-1.0, linestyle="--")
        ax[2].set_ylabel(r"$\tau / \tau_{\max}$")
        ax[2].set_title("Normalized Control Torque")
        ax[2].grid(True)


def plot(results: dict, dt=0.01, show_normalized_torque=True, sharey=False):
    """Plot results"""

    n_rows = 3 if show_normalized_torque else 2
    n_cols = len(results)

    fig, ax = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8 * n_cols, 8 if show_normalized_torque else 6),
        sharex=True,
        sharey="row" if sharey else False,
        squeeze=False,
    )

    for i, (controller_type, controller_results) in enumerate(results.items()):
        plot_results(
            ax=ax[:, i],
            controller_results=controller_results,
            dt=dt,
            show_normalized_torque=show_normalized_torque,
            controller_type=controller_type,
        )

    fig.tight_layout()
    plt.show()


def show_metrics(
    results: dict,
    dt=0.01,
    transient_time=5.0,
):
    metrics = {}

    for controller_type, controller_results in results.items():

        tau_cmd_hist = np.asarray(controller_results["tau_cmd_hist"])
        task_error_hists = controller_results["task_error_hists"]
        computation_times = np.asarray(controller_results["computation_times"])

        start_idx = int(transient_time / dt)

        # ------- Computation time -------
        comp_time_ms = np.mean(computation_times[start_idx:]) * 1e3

        # ------- Torque -------
        tau_rms = np.sqrt(np.mean(tau_cmd_hist[start_idx:] ** 2))

        controller_metrics = {
            "Computation time [ms]": comp_time_ms,
            "RMS torque [Nm]": tau_rms,
        }

        # ------- Tracking Performance -------
        for i, errors in enumerate(task_error_hists):
            errors = np.asarray(errors)

            if errors.size == 0:
                continue

            errors = errors[start_idx:]

            error_norm = np.linalg.norm(errors, axis=1)

            rmse = np.sqrt(np.mean(error_norm**2))
            max_error = np.max(np.abs(errors))

            controller_metrics[f"Task {i} RMSE [m]"] = rmse
            controller_metrics[f"Task {i} Max Error [m]"] = max_error

        metrics[controller_type] = controller_metrics

    # ------- Print table -------

    controller_names = list(metrics.keys())

    # Collect all metric names
    metric_names = []
    for controller_metrics in metrics.values():
        for metric in controller_metrics:
            if metric not in metric_names:
                metric_names.append(metric)

    # Column widths
    metric_width = 28
    value_width = 18

    header = f"{'Metric':<{metric_width}}"
    for name in controller_names:
        header += f"{name:>{value_width}}"

    print(header)
    print("-" * len(header))

    for metric in metric_names:
        row = f"{metric:<{metric_width}}"

        for controller in controller_names:
            value = metrics[controller].get(metric, np.nan)
            row += f"{value:>{value_width}.4f}"

        print(row)

    return metrics
