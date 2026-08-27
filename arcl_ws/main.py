import genesis as gs
from controller import TorqueControllerQP, TorqueControllerTraditional
from tasks import *
from robot import Robot
from pathlib import Path
import time

# use builtin panda 7dof robot with no hand
assets_dir = Path(gs.__file__).parent / "assets"
robot_xml_path = assets_dir / "xml/franka_emika_panda/panda_nohand.xml"

# simulation parameters
dt = 0.01
T = 10


def main():
    # initialize genesis
    gs.init(backend=gs.cpu)

    # create scene
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.81),
        ),
    )

    # add ground plane
    plane = scene.add_entity(gs.morphs.Plane())

    # create robot
    robot = Robot(
        scene=scene,
        robot_xml_path=robot_xml_path,
    )

    # define control objectives
    tasks = [
        XPositionTask(x_des=0.4),
        YPositionTask(y_des=0.4),
        ZSinusoidalTask(
            z_center=0.4,
            amplitude=0.15,
            frequency=0.1,
            dt=dt,
        ),
        # SphereTask(robot,.2,(.5,.5,.5),scene)
    ]

    # build scene
    scene.build()

    # set initial configuration
    torque_limits = robot.set_torque_limits(value=[20, 20, 20, 20, 12, 12, 12])
    robot.set_initial_pos(pos=(0.1, 0.5, 0.2), quat=(0, 1, 0, 0))
    time.sleep(2)

    # create controller object
    # controller = TorqueControllerQP(robot=robot, tasks=tasks)  # 22.5ms in first test run on my laptop
    controller = TorqueControllerTraditional(robot=robot, tasks=tasks) # 0.543ms in first test run on my laptop

    # simulate
    t = 0
    while t <= T:
        # get control torque
        control_torque = controller.compute_control_torque()

        # command control torque
        robot.command_torque(tau_cmd=control_torque)

        # step the simulation
        scene.step()

        # step time
        t += dt

    # computation time benchmark
    print(
        f"average controller computation time:\t{controller.average_time * 1000:.3f} ms"
    )

    # plot cartesian position error dynamics
    controller.plot_cartesian_pos_errors(torque_limits=torque_limits,dt=dt)


if __name__ == "__main__":
    main()
