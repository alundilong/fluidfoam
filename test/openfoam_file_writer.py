import sys
sys.path.insert(0,'../')

import numpy as np
from fluidfoam import readmesh, readscalar, readvector, OpenFoamFile, getVolumes

########################volScalarTypeField#############################
def writeScalarVolType(field, n_cells, bounfile, time, sol, field_name):
    boundary_names = list(bounfile.boundaryface.keys())
    content = internal_vol_scalar_field(field[0:n_cells],time,field_name)

    # boundaryfield
    content += "boundaryField\n{\n"
    start = n_cells
    end = start
    for name in boundary_names:
        key = name.decode('utf-8')
        bc_type = bounfile.boundaryface[str.encode(key)][b'type'].decode('utf-8')
        n_faces = int(bounfile.boundaryface[str.encode(key)][b'nFaces'])
        end += n_faces
        if 'empty' not in bc_type:
            #print(f"start:{start}, end:{end}")
            content += "\t"+f"{key}"+"\n\t{\n"
            content += "\t\ttype\t\tcalculated;\n"
            content += "\t\tvalue\t\tnonuniform List<scalar>\n"
            content += get_scalar_field(field[start:end])
            content += "\t}\n"
        else:
            content += "\t"+f"{key}"+"\n\t{\n"
            content += "\t\ttype\t\tempty;\n"
            content += "\t}\n"

        start = end

    content += "}\n"

    file_name = field_name+"_fluidfoam"
    with open(f"{sol}/{time}/{file_name}", "w") as file:
        file.write(content)

def internal_vol_scalar_field(numbers,time,field_name):
    # Create the base content with placeholders for numbers
    content = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  6
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;"""+f"\n\tlocation    \"{time}\";"+f"\n\tobject      {field_name}_fluidfoam;\n"+"""}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions  [0 0 0 0 0 0 0];

internalField   nonuniform List<scalar>
"""
    # Append the number of items and the list of numbers
    content += get_scalar_field(numbers)

    return content

def get_scalar_field(numbers):
    # Append the number of items and the list of numbers
    content = f"{len(numbers)}\n(\n"
    content += "\n".join(f"{num:.9f}" for num in numbers)
    content += "\n)\n;\n"

    return content


########################volVectorTypeField#############################
def writeVectorVolType(fieldx,fieldy,fieldz, n_cells, bounfile, time, sol, field_name):
    boundary_names = list(bounfile.boundaryface.keys())
    content = internal_vol_vector_field(fieldx[0:n_cells],fieldy[0:n_cells],fieldz[0:n_cells],time,field_name)

    # boundaryfield
    content += "boundaryField\n{\n"
    start = n_cells
    end = start
    for name in boundary_names:
        key = name.decode('utf-8')
        bc_type = bounfile.boundaryface[str.encode(key)][b'type'].decode('utf-8')
        n_faces = int(bounfile.boundaryface[str.encode(key)][b'nFaces'])
        end += n_faces
        if 'empty' not in bc_type:
            #print(f"start:{start}, end:{end}")
            content += "\t"+f"{key}"+"\n\t{\n"
            content += "\t\ttype\t\tcalculated;\n"
            content += "\t\tvalue\t\tnonuniform List<vector>\n"
            content += get_vector_field(fieldx[start:end],fieldy[start:end],fieldz[start:end])
            content += "\t}\n"
        else:
            content += "\t"+f"{key}"+"\n\t{\n"
            content += "\t\ttype\t\tempty;\n"
            content += "\t}\n"

        start = end

    content += "}\n"

    file_name = field_name+"_fluidfoam"
    with open(f"{sol}/{time}/{file_name}", "w") as file:
        file.write(content)

def internal_vol_vector_field(numbers_x,numbers_y,numbers_z,time,field_name):
    # Create the base content with placeholders for numbers
    content = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  6
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;"""+f"\n\tlocation    \"{time}\";"+f"\n\tobject      {field_name}_fluidfoam;\n"+"""}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions  [0 0 0 0 0 0 0];

internalField   nonuniform List<vector>
"""
    # Append the number of items and the list of numbers
    content += get_vector_field(numbers_x,numbers_y,numbers_z)

    return content

def get_vector_field(numbers_x,numbers_y,numbers_z):
    # Append the number of items and the list of numbers
    content = f"{len(numbers_x)}\n(\n"
    content += "\n".join(f"({x:.9f} {y:.9f} {z:.9f})" for x,y,z in zip(numbers_x, numbers_y, numbers_z))
    content += "\n)\n;\n"

    return content


########################volTensorTypeField#############################
def writeTensorVolType(fieldxx,fieldxy,fieldxz,\
        fieldyx,fieldyy,fieldyz,\
        fieldzx,fieldzy,fieldzz,\
        n_cells, bounfile, time, sol, field_name):
    boundary_names = list(bounfile.boundaryface.keys())
    content = internal_vol_tensor_field(fieldxx[0:n_cells],fieldxy[0:n_cells],fieldxz[0:n_cells],\
            fieldyx[0:n_cells],fieldyy[0:n_cells],fieldyz[0:n_cells],\
            fieldzx[0:n_cells],fieldzy[0:n_cells],fieldzz[0:n_cells],\
            time,field_name)

    # boundaryfield
    content += "boundaryField\n{\n"
    start = n_cells
    end = start
    for name in boundary_names:
        key = name.decode('utf-8')
        bc_type = bounfile.boundaryface[str.encode(key)][b'type'].decode('utf-8')
        n_faces = int(bounfile.boundaryface[str.encode(key)][b'nFaces'])
        end += n_faces

        if 'empty' not in bc_type:
            #print(f"start:{start}, end:{end}")
            content += "\t"+f"{key}"+"\n\t{\n"
            content += "\t\ttype\t\tcalculated;\n"
            content += "\t\tvalue\t\tnonuniform List<tensor>\n"
            content += get_tensor_field(fieldxx[start:end],fieldxy[start:end],fieldxz[start:end],\
                    fieldyx[start:end],fieldyy[start:end],fieldyz[start:end],\
                    fieldzx[start:end],fieldzy[start:end],fieldzz[start:end]\
                    )
            content += "\t}\n"
        else:
            content += "\t"+f"{key}"+"\n\t{\n"
            content += "\t\ttype\t\tempty;\n"
            content += "\t}\n"

        start = end

    content += "}\n"

    file_name = field_name+"_fluidfoam"
    with open(f"{sol}/{time}/{file_name}", "w") as file:
        file.write(content)

def internal_vol_tensor_field(n_xx,n_xy,n_xz,\
                              n_yx,n_yy,n_yz,\
                              n_zx,n_zy,n_zz,time,field_name):
    # Create the base content with placeholders for numbers
    content = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  6
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volTensorField;"""+f"\n\tlocation    \"{time}\";"+f"\n\tobject      {field_name}_fluidfoam;\n"+"""}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions  [0 0 0 0 0 0 0];

internalField   nonuniform List<tensor>
"""
    # Append the number of items and the list of numbers
    content += get_tensor_field(n_xx,n_xy,n_xz,\
                                n_yx,n_yy,n_yz,\
                                n_zx,n_zy,n_zz)

    return content

def get_tensor_field(n_xx,n_xy,n_xz,\
                     n_yx,n_yy,n_yz,\
                     n_zx,n_zy,n_zz):
    # Append the number of items and the list of numbers
    content = f"{len(n_xx)}\n(\n"
    content += "\n".join(f"({xx:.9f} {xy:.9f} {xz:.9f} {yx:.9f} {yy:.9f} {yz:.9f} {zx:.9f} {zy:.9f} {zz:.9f})" 
                         for xx,xy,xz,\
                             yx,yy,yz,\
                             zx,zy,zz \
                         in zip(n_xx,n_xy,n_xz,\
                                n_yx,n_yy,n_yz,\
                                n_zx,n_zy,n_zz))
    content += "\n)\n;\n"

    return content

########################surfaceScalarTypeField#############################
def writeScalarSurfaceType(field, n_internal_faces, bounfile, time, sol, field_name):
    boundary_names = list(bounfile.boundaryface.keys())
    content = internal_surface_scalar_field(field[0:n_internal_faces],time,field_name)

    # boundaryfield
    content += "boundaryField\n{\n"
    start = n_internal_faces
    end = start
    for name in boundary_names:
        key = name.decode('utf-8')
        bc_type = bounfile.boundaryface[str.encode(key)][b'type'].decode('utf-8')
        n_faces = int(bounfile.boundaryface[str.encode(key)][b'nFaces'])
        end += n_faces
        if 'empty' not in bc_type:
            #print(f"start:{start}, end:{end}")
            content += "\t"+f"{key}"+"\n\t{\n"
            content += "\t\ttype\t\tcalculated;\n"
            content += "\t\tvalue\t\tnonuniform List<scalar>\n"
            content += get_scalar_field(field[start:end])
            content += "\t}\n"
        else:
            content += "\t"+f"{key}"+"\n\t{\n"
            content += "\t\ttype\t\tempty;\n"
            content += "\t}\n"
        start = end

    content += "}\n"

    file_name = field_name+"_fluidfoam"
    with open(f"{sol}/{time}/{file_name}", "w") as file:
        file.write(content)

def internal_surface_scalar_field(numbers,time,field_name):
    # Create the base content with placeholders for numbers
    content = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  6
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       surfaceScalarField;"""+f"\n\tlocation    \"{time}\";"+f"\n\tobject      {field_name}_fluidfoam;\n"+"""}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions  [0 0 0 0 0 0 0];

internalField   nonuniform List<scalar>
"""
    # Append the number of items and the list of numbers
    content += get_scalar_field(numbers)

    return content

########################surfaceVectorTypeField#############################
def writeVectorSurfaceType(fieldx, fieldy, fieldz, n_internal_faces, bounfile, time, sol, field_name):
    boundary_names = list(bounfile.boundaryface.keys())
    content = internal_surface_vector_field(fieldx[0:n_internal_faces],fieldy[0:n_internal_faces],fieldz[0:n_internal_faces],time,field_name)

    # boundaryfield
    content += "boundaryField\n{\n"
    start = n_internal_faces
    end = start
    for name in boundary_names:
        key = name.decode('utf-8')
        bc_type = bounfile.boundaryface[str.encode(key)][b'type'].decode('utf-8')
        n_faces = int(bounfile.boundaryface[str.encode(key)][b'nFaces'])
        end += n_faces
        if 'empty' not in bc_type:
            #print(f"start:{start}, end:{end}")
            content += "\t"+f"{key}"+"\n\t{\n"
            content += "\t\ttype\t\tcalculated;\n"
            content += "\t\tvalue\t\tnonuniform List<vector>\n"
            content += get_vector_field(fieldx[start:end],fieldy[start:end],fieldz[start:end])
            content += "\t}\n"
        else:
            content += "\t"+f"{key}"+"\n\t{\n"
            content += "\t\ttype\t\tempty;\n"
            content += "\t}\n"
        start = end

    content += "}\n"

    file_name = field_name+"_fluidfoam"
    with open(f"{sol}/{time}/{file_name}", "w") as file:
        file.write(content)

def internal_surface_vector_field(numbers_x,numbers_y,numbers_z,time,field_name):
    # Create the base content with placeholders for numbers
    content = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  6
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       surfaceVectorField;"""+f"\n\tlocation    \"{time}\";"+f"\n\tobject      {field_name}_fluidfoam;\n"+"""}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions  [0 0 0 0 0 0 0];

internalField   nonuniform List<vector>
"""
    # Append the number of items and the list of numbers
    content += get_vector_field(numbers_x, numbers_y, numbers_z)

    return content
