import genesis as gs
from robot import Robot
from controller import TorqueControllerQP
from pathlib import Path
import numpy as np
import time


def run(
    tasks,
    controller=TorqueControllerQP(),
    show_viewer=False,
    dt=0.01,
    T=10,
    gravity=(0.0, 0.0, -9.81),
    pos0=(0, 0, 1),
    quat0=None,
    q0=None,
    torque_limits=None,
    robot_xml_path=Path(gs.__file__).parent
    / "assets/xml/franka_emika_panda/panda_nohand.xml",
):
    """runs the simulation loop with specified parameters

    Args:
        tasks (list): List of control objectives in order of priority.
        controller (TorqueController, optional): Torque controller object to control the robot. Defaults to TorqueControllerQP().
        show_viewer (bool, optional): Whether to show the genesis viewer or not. Defaults to False.
        dt (float, optional): Simulation time step in seconds. Defaults to 0.01.
        T (float, optional): Simulation runtime in seconds. Defaults to 10.
        gravity (tuple, optional): Gravity vector. Defaults to (0.0, 0.0, -9.81).
        pos0 (tuple, optional): Initial robot ee position. Defaults to (0,0,1).
        quat0 (tuple, optional): Initial robot ee quaternion. Defaults to None.
        q0 (tuple, optional): Initial robot joint configuration. Defaults to None.
        torque_limits (tuple, optional): Artificial torque limits for the joints. Defaults to None.
        robot_xml_path (Path, optional): Path to the robots models xml file. Defaults to the builtin genesis franka emika panda 7dof with no hand.

    Returns:
        tau_cmd_hist, tasks, torque_limits
    """
    # create scene
    scene = gs.Scene(
        show_viewer=show_viewer,
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=gravity,
        ),
    )

    # add ground plane
    _ = scene.add_entity(gs.morphs.Plane())

    # create robot
    robot = Robot(
        scene=scene,
        robot_xml_path=robot_xml_path,
        gravity=gravity,
    )

    # build scene
    scene.build()

    # set initial configuration
    torque_limits = robot.retrieve_torque_limits(torque_limits=torque_limits)
    if q0 is not None:
        robot.set_qpos(qpos=q0)
    else:
        robot.set_pos(pos=pos0, quat=quat0)

    # let scene settle
    for _ in range(10):
        scene.step()

    # simulate
    tau_cmd_hist = []
    computation_times = []
    t = 0
    while t <= T:
        # read robot state
        state = robot.get_state()

        # get control torque
        tic = time.perf_counter()
        tau_cmd = controller.compute_control_torque(
            state=state,
            tasks=tasks,
            torque_limits=torque_limits,
        )
        toc = time.perf_counter()
        computation_times.append(toc - tic)
        tau_cmd_hist.append(tau_cmd)

        # command control torque
        robot.command_torque(tau_cmd=tau_cmd)

        # step the simulation
        scene.step()

        # step time
        t += dt

    # extract task errors for plots and reset the tasks
    task_error_hists = []
    for task in tasks:
        task_error_hists.append(task.get_errors())
        task.reset()

    # convert command torques for plots
    tau_cmd_hist = np.array(tau_cmd_hist)

    # reset the simulation
    scene.destroy()

    return {
        "tau_cmd_hist": tau_cmd_hist,
        "task_error_hists": task_error_hists,
        "computation_times": computation_times,
        "torque_limits": torque_limits,
    }
