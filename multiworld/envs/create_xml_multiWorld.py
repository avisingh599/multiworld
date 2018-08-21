import numpy as np
import xml.etree.cElementTree as ET
import xml.dom.minidom as minidom
import imp
import glob
import os
import random


import numpy as np
import stl
from stl import mesh



def find_mins_maxs(obj):
    minx = maxx = miny = maxy = minz = maxz = None
    for p in obj.points:
        # p contains (x, y, z)
        if minx is None:
            minx = p[stl.Dimension.X]
            maxx = p[stl.Dimension.X]
            miny = p[stl.Dimension.Y]
            maxy = p[stl.Dimension.Y]
            minz = p[stl.Dimension.Z]
            maxz = p[stl.Dimension.Z]
        else:
            maxx = max(p[stl.Dimension.X], maxx)
            minx = min(p[stl.Dimension.X], minx)
            maxy = max(p[stl.Dimension.Y], maxy)
            miny = min(p[stl.Dimension.Y], miny)
            maxz = max(p[stl.Dimension.Z], maxz)
            minz = min(p[stl.Dimension.Z], minz)


    return minx, maxx, miny, maxy, minz, maxz


def file_len(fname):
    i = 0
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1




ASSET_BASE_DIR = os.path.join(os.path.dirname(__file__), 'assets/meshes/')


def create_object_xml(num_objects, object_mass, friction_params, object_meshes,
                      maxlen, minlen, reset_xml, obj_classname = None,
                      block_height = 0.03, block_width = 0.03):
    """
    :param hyperparams:
    :param load_dict_list: if not none load configuration, instead of sampling
    :return: if not loading, save dictionary with static object properties
    """
    #xmldir = '/'.join(str.split(filename, '/')[:-1])
    root = ET.Element("top")

    save_dict_list = []

  
    f_sliding, f_torsion, f_rolling = friction_params
    world_body = ET.SubElement(root, "worldbody")

    loaded_meshes = {}

    if reset_xml is not None:
        load_dict_list = reset_xml
    else: load_dict_list = None

    for i in range(num_objects):
        if load_dict_list == None:
            dict = {}

            color1 = dict['color1'] = np.random.uniform(0.3, 1., 3)
            color2 = dict['color2'] = np.random.uniform(0.3, 1., 3)


            l1 = dict['l1'] =np.random.uniform(minlen, maxlen)
            l2 = dict['l2'] =np.random.uniform(minlen, maxlen)

            pos2 = dict['pos2']= np.random.uniform(0.01, l1)

            if object_meshes is not None:
                dict['chosen_mesh'] = chosen_mesh = random.choice(object_meshes)

        else:
            dict = load_dict_list[i]
            color1 = dict['color1']
            color2 = dict['color2']
            l1 = dict['l1']
            l2 = dict['l2']
            pos2 = dict['pos2']
            chosen_mesh = dict['chosen_mesh']
        save_dict_list.append(dict)

        obj_string = "object{}".format(i)
        print('using friction=({}, {}, {}), object mass{}'.format(f_sliding, f_torsion, f_rolling, object_mass))
        if object_meshes is not None:
            assets = ET.SubElement(root, "asset")
            if chosen_mesh not in loaded_meshes:
                o_mesh = ASSET_BASE_DIR + '{}/'.format(chosen_mesh)
                print('import mesh dir', o_mesh)
                stl_files = glob.glob(o_mesh + '*.stl')
                convex_hull_files = [x for x in stl_files if 'Shape_IndexedFaceSet' in x]
                object_file = [x for x in stl_files
                               if x not in convex_hull_files and 'Lamp' not in x and 'Camera' not in x and 'GM' not in x][0]



                mesh_object = mesh.Mesh.from_file(object_file)
                minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(mesh_object)


                if chosen_mesh in ['Knife', 'Fork', 'Spoon']:      #should add a more extensible way to handle different rescale rules
                    max_length = max((maxx - minx), (maxy - miny))
                    scale = [2 * maxlen / max_length for _ in range(3)]

                    if chosen_mesh == 'Knife':
                        scale[2] *= 10
                elif chosen_mesh in ['Bowl', 'ElephantBowl', 'GlassBowl', 'LotusBowl01', 'RuggedBowl']:
                    min_length = min((maxx - minx), (maxy - miny))
                    scale = [maxlen / min_length for _ in range(3)]
                else:
                    max_length = max(max((maxx - minx), (maxy - miny)), (maxz - minz))
                    scale = [maxlen / max_length for _ in range(2)] + [maxlen / (maxz - minz) * 0.75]

                # object_pos = [0., 0., 0.]
                # object_pos[0] -= scale[0] * (minx + maxx) / 2.0
                # object_pos[1] -= scale[1] * (miny + maxy) / 2.0
                # object_pos[2] -= 0.08 + scale[2] * (minz + maxz) / 2.0

                object_pos = [0, 0.7, 0]

                mass_per_elem, n_cvx_files = object_mass / (1 + len(convex_hull_files)), len(convex_hull_files)
                loaded_meshes[chosen_mesh] = (object_pos, mass_per_elem, n_cvx_files)


                ET.SubElement(assets, "mesh", name=chosen_mesh + "_mesh", file=object_file.strip('assets/meshes/'),
                              scale="{} {} {}".format(scale[0], scale[1], scale[2]))
                for n, c_file in enumerate(convex_hull_files):
                    ET.SubElement(assets, "mesh", name=chosen_mesh + "_convex_mesh{}".format(n), file=c_file.strip('assets/meshes/'),
                                  scale="{} {} {}".format(scale[0], scale[1], scale[2]))

            else: object_pos, mass_per_elem, n_cvx_files = loaded_meshes[chosen_mesh]

            pos_str = "{} {} {}".format(object_pos[0], object_pos[1], object_pos[2])
            angle_str = "{} {} {}".format(0, 0, np.random.uniform(0, 6.28))


            
            ET.SubElement(world_body, 'include', file = 'sawyer_xyz_base.xml')

          

            ET.SubElement(world_body, 'site', name = 'goal', pos = '0.3 0.9 0.02', size = '0.02', rgba='1 0.5 0.5 0.5')

            if obj_classname is not None:
                obj = ET.SubElement(world_body, "body",name=obj_string, pos=pos_str, euler=angle_str,
                                    childclass=obj_classname)
            else: obj = ET.SubElement(world_body, "body",name=obj_string, pos=pos_str, euler=angle_str)

            ET.SubElement(obj, "joint", type="free", limited='false')
            ET.SubElement(obj, 'inertial', pos='0 0 0', mass=str(object_mass), diaginertia='10000 10000 10000')

            #visual mesh
            ET.SubElement(obj, "geom", type="mesh", mesh = chosen_mesh + "_mesh",
                          rgba="{} {} {} 1".format(color1[0], color1[1], color1[2]), mass="{}".format(mass_per_elem),
                          contype="0", conaffinity="0", name='objGeom')
            #contact meshes
            for n in range(n_cvx_files):
                ET.SubElement(obj, "geom", type="mesh", mesh=chosen_mesh + "_convex_mesh{}".format(n),
                              rgba="0 1 0 0", mass="{}".format(mass_per_elem),
                              contype="1", conaffinity="1", friction="{} {} {}".format(f_sliding, f_torsion, f_rolling)
                              )

        else:
            obj = None
            if obj_classname is not None:
                obj = ET.SubElement(world_body, "body", name=obj_string, pos="0 0 0",
                                    childclass=obj_classname)
            else: obj = ET.SubElement(world_body, "body", name=obj_string, pos="0 0 0")

            ET.SubElement(obj, "joint", type="free", limited='false')
            ET.SubElement(obj, 'inertial', pos='0 0 0', mass='1', diaginertia='10000 10000 10000')

            ET.SubElement(obj, "geom", type="box", size="{} {} {}".format(block_width, l1, block_height),
                          rgba="{} {} {} 1".format(color1[0], color1[1], color1[2]), mass="{}".format(object_mass),
                          contype="1", conaffinity="1", friction="{} {} {}".format(f_sliding, f_torsion, f_rolling)
                          )
            ET.SubElement(obj, "geom", pos="{} {} 0.0".format(l2, pos2),
                          type="box", size="{} {} {}".format(l2, block_width, block_height),
                          rgba="{} {} {} 1".format(color2[0], color2[1], color2[2]), mass="{}".format(object_mass),
                          contype="1", conaffinity="1", friction="{} {} {}".format(f_sliding, f_torsion, f_rolling)
                          )


        # if sensor_frame is None:
        #     sensor_frame = ET.SubElement(root, "sensor")
        #ET.SubElement(sensor_frame, "framepos", name=obj_string + '_sensor', objtype="body", objname=obj_string)

    tree = ET.ElementTree(root)

    xml_str = minidom.parseString(ET.tostring(
        tree.getroot(),
        'utf-8')).toprettyxml(indent="    ")

    

    xml_str = xml_str.splitlines()[2: -1]
    xml_str = "\n".join(xml_str).lstrip(' ')

    return xml_str



def create_xml(fileName = 'temp', objectMass = 0.5, friction_params = [1, 0.1, 0.02], objMesh = 'fox', block_height = None, block_width = None, maxlen = 0.2, minlen = 0.1):

    if objMesh == None:
        assert block_width and block_height

    from jinja2 import Environment, FileSystemLoader
    file_loader = FileSystemLoader( os.path.join(os.path.dirname(__file__), 'assets/templates'))

    env = Environment(loader=file_loader)

    template = env.get_template('pickPlace.xml')

    objectXML_str = create_object_xml( num_objects = 1, object_mass = objectMass, friction_params = friction_params , object_meshes = [objMesh],
                          maxlen = maxlen, minlen = minlen, reset_xml = None, obj_classname = None,
                          block_height = block_height, block_width = block_width)


    output = template.render(objectData=objectXML_str)

    outfile = os.path.join(os.path.dirname(__file__), 'assets/sawyer_xyz')+'/'+fileName+'.xml'



    with open(outfile, 'w') as f:
      
        f.write(output)

    print(output)

#create(maxlen = 0.1, objMesh = 'Fork')


for maxlen in [0.10]:

    for objMesh in ['Bowl', 'cupcake', 'Elephant', 'ElephantBowl', 'Fork', 'fox', 'GlassBowl', 'Knife', 'Pawn', 'Queen', 'Rook', 'Spoon', 'ServingBowl']:
    #for objMesh in ['Fork']:
        create_xml(maxlen = maxlen, objMesh = objMesh, fileName = 'temp_'+objMesh+str(maxlen))