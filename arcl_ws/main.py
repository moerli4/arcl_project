import genesis as gs
from controller import TorqueController
from robot import Robot
import pinocchio as pin
from pathlib import Path

assets_dir = Path(gs.__file__).parent / "assets"
robot_xml_path = assets_dir / "xml/franka_emika_panda/panda_nohand.xml" # use builtin panda 7dof robot with no hand

def main():
    # initialize genesis
    gs.init(backend=gs.gpu)

    # create scene
    scene = gs.Scene(
        show_viewer=True,
        sim_options=gs.options.SimOptions(
            dt=0.01,
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

    # build scene
    scene.build()

    # create controller object
    controller = TorqueController(
        robot=robot,
    )

    # simulate
    while True:
        # get control torque
        control_torque = controller.compute_control_torque()

        # command control torque
        robot.command_torque(tau_cmd=control_torque)

        # step the simulation
        scene.step()

if __name__ == "__main__":
    main()
