import yaml
filename = "meshes/mesh_size_5000/meshstatistic.yml"

with open(filename) as meshstats:
    ms = yaml.load(meshstats, Loader=yaml.UnsafeLoader)
    print(ms['ecs_share'])
    print("\n")

    for key in ms.keys():
        #print(key, " : ",  ms[key], "\n")

        print("surface_to_vol_astro", ms['cell_surface'][-1]/ms['cell_volume'][-1]*1000)
        print("surface_to_vol_neuro", sum(ms['cell_surface'])/sum(ms['cell_volume'])*1000)

        if key == "cell_volume":
            print("vol glial", ms[key][-1]*1.0e-9, "um^3")
            print("vol neuron", sum((ms[key][0:-1]))*1.0e-9, "um^3")
            print("vol all cells", sum((ms[key]))*1.0e-9, "um^3")
            vol_g = ms[key][-1]*1.0e-9
            vol_n = sum((ms[key][0:-1]))*1.0e-9
            vol_a = sum((ms[key]))*1.0e-9
        if key == "ecs_volume":
            print("vol ecs", ms[key]*1.0e-9, "um^3")
            vol_e = ms[key]*1.0e-9

    print(vol_g + vol_n + vol_e)
    print(vol_a + vol_e)

    print(vol_e / (vol_a + vol_e))
    print(vol_g / (vol_a + vol_e))
    print(vol_n / (vol_a + vol_e))


    for key in ms.keys():
        #print(key, " : ",  ms[key], "\n")
        #print(ms['cell_surface'][-2]/ms['cell_volume'][-2]*1000)
        if key == "cell_surface":
            print("surf glial", ms[key][-1]*1.0e-6, "um^2")
            print("surf neuron", sum((ms[key][0:-1]))*1.0e-6, "um^2")
            print("surf all cells", sum((ms[key]))*1.0e-6, "um^2")
            vol_g = ms[key][-1]*1.0e-6
            vol_n = sum((ms[key][0:-1]))*1.0e-6
            vol_a = sum((ms[key]))*1.0e-6

    print(vol_g + vol_n)
    print(vol_a + vol_e)

    print("per g", vol_g / (vol_a)*100)
    print("per n", vol_n / (vol_a)*100)
