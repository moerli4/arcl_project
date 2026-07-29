import genesis as gs

def main():
    # initialize genesis
    gs.init(backend=gs.gpu)

    # create scene
    scene = gs.Scene(
        show_viewer=True,
        sim_options=gs.options.SimOptions(
            dt=0.002,
            gravity=(0.0, 0.0, -9.81),
        ),
    )
    
    # add ground plane
    plane = scene.add_entity(gs.morphs.Plane())

    # add franka robot (use builtin panda model)
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    # build scene
    scene.build()

    # simulate
    while True:
        scene.step()

if __name__ == "__main__":
    main()
