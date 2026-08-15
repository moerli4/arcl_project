import genesis as gs
from controller import TorqueController
from tasks import *
from robot import Robot
from pathlib import Path
import time

assets_dir = Path(gs.__file__).parent / "assets"
robot_xml_path = (
    assets_dir / "xml/franka_emika_panda/panda_nohand.xml"
)  # use builtin panda 7dof robot with no hand
dt = 0.01


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
        # SphereTask(robot=robot,radius=.2,center=(.4,.4,.4), scene=scene),
        # CylinderTask(robot=robot,radius=.5,center=(.8,.8), scene=scene),
        # PointToPointTask(p0=(.4,.7,.4), p1=(.7,.4,.7), period=40, scene=scene, dt=dt),
        HorizontalPlaneTask(height=0.2),
        MaximizeManipulabilityTask(robot=robot),
    ]

    # build scene
    scene.build()

    # set initial configuration
    robot.set_initial_pos(pos=(0.4, 0.4, 0.2), quat=(0, 1, 0, 0))
    time.sleep(2)

    # create controller object
    controller = TorqueController(robot=robot, tasks=tasks)

    # simulate
    while True:
        # get control torque
        control_torque = controller.compute_control_torque()

        # command control torque
        robot.command_torque(tau_cmd=control_torque)

        print(robot.get_state()["q"])

        # step the simulation
        scene.step()


if __name__ == "__main__":
    main()
