import genesis as gs
from controller import TorqueController

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

    # add franka robot (use builtin panda model)
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml",
        pos   = (0.0, 0.0, 0.0),
        euler = (0, 0, 0),
        ),
        # material=gs.materials.Rigid(
        #     gravity_compensation=1.0,   # compensate gravity internally
        # ),
    )
    arm_jnt_names = ['joint1','joint2','joint3','joint4','joint5','joint6','joint7']
    franka_arm_dofs_idx = [franka.get_joint(name).dof_idx_local for name in arm_jnt_names]

    # build scene
    scene.build()

    # create controller object
    controller = TorqueController(
        robot=franka,
        robot_dofs_idx=franka_arm_dofs_idx,
        end_effector_link_name="hand",
    )

    # simulate
    while True:
        # get control torque
        control_torque = controller.compute_control_torque()

        # command control torque
        franka.control_dofs_force(
            control_torque,
            franka_arm_dofs_idx,
        )

        # step the simulation
        scene.step()

if __name__ == "__main__":
    main()
