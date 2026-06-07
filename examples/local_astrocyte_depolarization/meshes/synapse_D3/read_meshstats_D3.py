import yaml
filename = "meshes/meshstatistic.yml"

with open(filename) as meshstats:
    ms = yaml.load(meshstats, Loader=yaml.UnsafeLoader)
    print(ms['ecs_share'])
    print("\n")

    print("surface_to_vol_astro", ms['cell_surface'][0]/ms['cell_volume'][0]*1000)
    print("surface_to_vol_neuro", sum(ms['cell_surface'][1:])/sum(ms['cell_volume'][1:])*1000)

    for key in ms.keys():
        if key == "cell_volume":

            vol_g = ms[key][0]*1.0e-9
            vol_n = sum((ms[key][1:]))*1.0e-9
            vol_a = sum((ms[key]))*1.0e-9

            print("---------------------------")
            print("Volumes:")
            print("vol glial", vol_g, "um^3")
            print("vol neuron", vol_n, "um^3")
            print("vol all cells", vol_a, "um^3")
            print("vol all cells", vol_g + vol_n, "um^3")
            print("---------------------------")
 
        if key == "ecs_volume":
            print("vol ecs", ms[key]*1.0e-9, "um^3")
            vol_e = ms[key]*1.0e-9

    print("volume total:", vol_g + vol_n + vol_e)
    print("volume total:", vol_a + vol_e)

    print("---------------------------")
    print("Sanity check:")
    print("ECS vol share", vol_e / (vol_a + vol_e)*100)
    print("glial vol share", vol_g / (vol_a + vol_e)*100)
    print("neuorn vol share", vol_n / (vol_a + vol_e)*100)

    print("sum", vol_e / (vol_a + vol_e)*100
               + vol_g / (vol_a + vol_e)*100
               + vol_n / (vol_a + vol_e)*100)
    print("---------------------------")

    for key in ms.keys():
        if key == "cell_surface":

            surf_g = ms[key][0]*1.0e-6
            surf_n = sum((ms[key][1:]))*1.0e-6
            surf_a = sum((ms[key]))*1.0e-6

            print("surf all cells", surf_g + surf_n, "um^2")
            print("---------------------------")
            print("Surfaces:")
            print("surf glial no", "um^2")
            print("surf glial", surf_g, "um^2")
            print("surf neuron", surf_n, "um^2")
            print("surf all cells", sum((ms[key]))*1.0e-6, "um^2")
            print("surf all cells", surf_g + surf_n, "um^2")
            print("---------------------------")


    print("---------------------------")
    print("Sanity check:")
    print(vol_g + vol_n + vol_e)
    print(vol_a + vol_e)
    print("---------------------------")

    print("---------------------------")
    print("Sanity check:")
    print("volume g per all", vol_g / (vol_a + vol_e)*100)
    print("volume n per all", vol_n / (vol_a + vol_e)*100)
    print("---------------------------")
