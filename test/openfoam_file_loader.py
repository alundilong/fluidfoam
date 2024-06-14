import sys
sys.path.insert(0,'../')

import numpy as np
from fluidfoam import readmesh, readscalar, readvector, OpenFoamFile, getVolumes

########################volScalarTypeField#############################
def readScalarVolType(field, n_cells, bounfile, time, sol, field_name):
    boundary_names = list(bounfile.boundaryface.keys())
    internal = readscalar(sol, time, field_name, verbose=False)
    if(len(internal) == 1):
        #print("uniform field")
        internal = np.tile(internal, n_cells)
        #print(type(internal),internal.shape)
    field[0:n_cells] = internal

    # boundaryfield
    start = n_cells
    end = start
    for name in boundary_names:
        key = name.decode('utf-8')
        n_start_face = int(bounfile.boundaryface[str.encode(key)][b'startFace'])
        n_faces = int(bounfile.boundaryface[str.encode(key)][b'nFaces'])
        boundaryfield = readscalar(sol,time,field_name,boundary=key,verbose=False)
        if(len(boundaryfield) == 1):
            print("uniform field")
            boundaryfield = np.tile(boundaryfield, n_faces)
            print(type(boundaryfield),boundaryfield.shape)
    
        end += n_faces
        field[start:end] = boundaryfield
        #print(f"start:{start}, end:{end}")
        start = end

########################volVectorTypeField#############################
def readVectorVolType(fieldx,fieldy,fieldz, n_cells, bounfile, time, sol, field_name):
    boundary_names = list(bounfile.boundaryface.keys())
    internalx,internaly,internalz = readvector(sol, time, field_name, verbose=False)
    if(len(internalx) == 1):
        #print("uniform field")
        internalx = np.tile(internalx, n_cells)
        internaly = np.tile(internaly, n_cells)
        internalz = np.tile(internalz, n_cells)
        #print(type(internalx),internalx.shape)
        #print(type(internaly),internaly.shape)
        #print(type(internalz),internalz.shape)
    fieldx[0:n_cells] = internalx
    fieldy[0:n_cells] = internaly
    fieldz[0:n_cells] = internalz

    # boundaryfield
    start = n_cells
    end = start
    for name in boundary_names:
        key = name.decode('utf-8')
        n_start_face = int(bounfile.boundaryface[str.encode(key)][b'startFace'])
        n_faces = int(bounfile.boundaryface[str.encode(key)][b'nFaces'])
        boundaryfieldx,boundaryfieldy,boundaryfieldz = readvector(sol,time,field_name,boundary=key,verbose=False)
        if(len(boundaryfieldx) == 1):
            #print("uniform field")
            boundaryfieldx = np.tile(boundaryfieldx, n_faces)
            boundaryfieldy = np.tile(boundaryfieldy, n_faces)
            boundaryfieldz = np.tile(boundaryfieldz, n_faces)
            #print(type(boundaryfieldx),boundaryfieldx.shape)
            #print(type(boundaryfieldy),boundaryfieldy.shape)
            #print(type(boundaryfieldz),boundaryfieldz.shape)
    
        end += n_faces
        fieldx[start:end] = boundaryfieldx
        fieldy[start:end] = boundaryfieldy
        fieldz[start:end] = boundaryfieldz
        #print(f"start:{start}, end:{end}")
        start = end

########################volScalarTypeField#############################
def readScalarSurfaceType(field, n_internal_faces, bounfile, time, sol, field_name):
    boundary_names = list(bounfile.boundaryface.keys())
    # internalfield
    phi_i = readscalar(sol,time,field_name,verbose=False)
    if len(phi_i) == 1:
        #print("uniform field")
        phi_i = np.tile(phi_i,  n_internal_faces)
        #print(type(phi_i),phi_i.shape)
    #else:
        #print("nonuniform field")
        #print(type(phi_i),phi_i.shape)
    field[0:n_internal_faces] = phi_i
    
    # boundaryfield
    start = n_internal_faces
    end = start
    for name in boundary_names:
        key = name.decode('utf-8')
        n_start_face = int(bounfile.boundaryface[str.encode(key)][b'startFace'])
        n_faces = int(bounfile.boundaryface[str.encode(key)][b'nFaces'])
        phi_b = readscalar(sol,time,field_name,boundary=key,verbose=False)
        #print(key)
        if len(phi_b) == 1:
            #print("uniform field")
            phi_b = np.tile(phi_b, n_faces)
            #print(type(phi_b),phi_b.shape)
        #else:
            #print("nonuniform field")
            #print(type(phi_b),phi_b.shape)
    
        end += n_faces
        field[start:end] = phi_b
        start = end

########################volVectorTypeField#############################
def readVectorSurfaceType(fieldx,fieldy,fieldz, n_internal_faces, bounfile, time, sol, field_name):
    boundary_names = list(bounfile.boundaryface.keys())
    # internalfield
    phi_ix,phi_iy,phi_iz = readvector(sol,time,field_name,verbose=False)
    if len(phi_ix) == 1:
        #print("uniform field")
        phi_ix = np.tile(phi_ix,  n_internal_faces)
        phi_iy = np.tile(phi_iy,  n_internal_faces)
        phi_iz = np.tile(phi_iz,  n_internal_faces)
        #print(type(phi_ix),phi_ix.shape)
        #print(type(phi_iy),phi_iy.shape)
        #print(type(phi_iz),phi_iz.shape)
    #else:
        #print("nonuniform field")
        #print(type(phi_ix),phi_ix.shape)
    fieldx[0:n_internal_faces] = phi_ix
    fieldy[0:n_internal_faces] = phi_iy
    fieldz[0:n_internal_faces] = phi_iz
    
    # boundaryfield
    start = n_internal_faces
    end = start
    for name in boundary_names:
        key = name.decode('utf-8')
        n_start_face = int(bounfile.boundaryface[str.encode(key)][b'startFace'])
        n_faces = int(bounfile.boundaryface[str.encode(key)][b'nFaces'])
        phi_bx,phi_by,phi_bz = readvector(sol,time,field_name,boundary=key,verbose=False)
        #print(key)
        if len(phi_bx) == 1:
            #print("uniform field")
            phi_bx = np.tile(phi_bx, n_faces)
            phi_by = np.tile(phi_by, n_faces)
            phi_bz = np.tile(phi_bz, n_faces)
            #print(type(phi_bx),phi_bx.shape)
            #print(type(phi_by),phi_by.shape)
            #print(type(phi_bz),phi_bz.shape)
        #else:
            #print("nonuniform field")
            #print(type(phi_bx),phi_bx.shape)
            #print(type(phi_by),phi_by.shape)
            #print(type(phi_bz),phi_bz.shape)
    
        end += n_faces
        fieldx[start:end] = phi_bx
        fieldy[start:end] = phi_by
        fieldz[start:end] = phi_bz
        start = end

if __name__ == "__main__":
    sol = '../../../cases/damBreak_simple/'
    time = "0.001"
    field_name = "U"
    key="leftWall"
    boundaryfieldx,boundaryfieldy,boundaryfieldz = readvector(sol,time,field_name,boundary=key,verbose=True)
