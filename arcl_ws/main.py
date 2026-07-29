import genesis as gs
from controller import TorqueController
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

    # add robot to the scene
    robot = scene.add_entity(
        gs.morphs.MJCF(file=robot_xml_path,
        pos   = (0.0, 0.0, 0.0),
        euler = (0, 0, 0),
        ),
    )
    arm_jnt_names = ['joint1','joint2','joint3','joint4','joint5','joint6','joint7']
    robot_arm_dofs_idx = [robot.get_joint(name).dof_idx_local for name in arm_jnt_names]

    # create pinocchio robot model
    pin_robot_model = pin.buildModelsFromMJCF(filename=robot_xml_path)[0]
    data = pin_robot_model.createData()

    # build scene
    scene.build()

    # create controller object
    controller = TorqueController(
        robot=robot,
        robot_dofs_idx=robot_arm_dofs_idx,
        pin_model=pin_robot_model,
        pin_data=data,
        end_effector_link_name="link7",
    )

    # simulate
    while True:
        # get control torque
        control_torque = controller.compute_control_torque()

        # command control torque
        robot.control_dofs_force(
            control_torque,
            robot_arm_dofs_idx,
        )

        # step the simulation
        scene.step()

if __name__ == "__main__":
    main()
