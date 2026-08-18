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
    gs.init(backend=gs.gpu)

    # create scene
    scene = gs.Scene(
        show_viewer=True,
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

        # HorizontalPlaneTask(height=0.2),
        # PointSequenceTask(
        #     points=[(0.1, 0.5, 0.2),(0.5, 0.5, 0.2),(0.5, 0.1, 0.4)],
        #     segment_time=10.0,
        #     scene=scene,
        #     dt=dt,
        # ),

        # MaximizeManipulabilityTask(robot=robot),
        # AvoidJointLimitsTask(robot=robot),
    ]

    # build scene
    scene.build()

    # set initial configuration
    robot.set_torque_limits(value=1)
    robot.set_initial_pos(pos=(0.1, 0.5, 0.2), quat=(0, 1, 0, 0))
    time.sleep(2)

    # create controller object
    # controller = TorqueControllerQP(robot=robot, tasks=tasks)
    controller = TorqueControllerTraditional(robot=robot, tasks=tasks)

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

    # plot cartesian position error dynamics
    controller.plot_cartesian_pos_errors(dt=dt)


if __name__ == "__main__":
    main()
