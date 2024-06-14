import sys, os, re
sys.path.insert(0,'../')

import numpy as np
from fluidfoam import readmesh, readscalar, readvector, OpenFoamFile, getVolumes
from openfoam_file_loader import readScalarVolType, readScalarSurfaceType, readVectorVolType, readVectorSurfaceType
from openfoam_file_writer import writeScalarVolType, writeVectorVolType, writeTensorVolType,writeScalarSurfaceType
from openfoam_file_writer import writeScalarSurfaceType, writeVectorSurfaceType #, writeTensorSurfaceType

#####################################################
sol = '../../../../cases/damBreak_simple/'
#sol = '../../../cases/damBreak_simple2/'
dt = 0.001

c, vols = getVolumes(sol,verbose=False)
total_vol_array_size = 0
# internal cells
xs,ys,zs = readmesh(sol,verbose=False)
print(len(xs),len(ys),len(zs))
n_cells = len(xs)

total_vol_array_size += n_cells

# boundary mesh
bounfile = OpenFoamFile(sol+"/constant/polyMesh",name="boundary",verbose=False)
print(bounfile.boundaryface)
boundary_names = list(bounfile.boundaryface.keys())
for name in boundary_names:
    key = name.decode('utf-8')
    xs,ys,zs = readmesh(sol,boundary=key)
    n_start_face = int(bounfile.boundaryface[str.encode(key)][b'startFace'])
    n_faces = int(bounfile.boundaryface[str.encode(key)][b'nFaces'])
    total_vol_array_size += n_faces
    print(f'{key}: {len(xs)} {len(ys)} {len(zs)}, startFace:{n_start_face}, nFaces:{n_faces}')

ownerfile = OpenFoamFile(sol+"/constant/polyMesh",name="owner",verbose=False)
print(ownerfile.nb_cell, ownerfile.nb_faces)
print(len(ownerfile.values))

neighfile = OpenFoamFile(sol+"/constant/polyMesh",name="neighbour",verbose=False)
n_internal_faces = neighfile.nb_faces
print(neighfile.nb_cell, n_internal_faces)
print(len(neighfile.values))

total_surface_array_size = len(ownerfile.values)

Cx = np.zeros(total_vol_array_size)
Cy = np.zeros(total_vol_array_size)
Cz = np.zeros(total_vol_array_size)
readVectorVolType(Cx,Cy,Cz, n_cells, bounfile, "constant/polyMesh", sol, "C")
print("-"*50+"(C)")
print(Cx.shape, Cy.shape, Cz.shape)

# internal field of magSf
magSf = np.zeros(total_surface_array_size)
readScalarSurfaceType(magSf, n_internal_faces, bounfile, "constant/polyMesh", sol, "magSf")
print("-"*50+"(magSf)")
print(magSf.shape)

weights = np.zeros(total_surface_array_size)
readScalarSurfaceType(weights, n_internal_faces, bounfile, "constant", sol, "weights")
print("-"*50+"(weights)")
print(weights.shape)

deltaCoeffs = np.zeros(total_surface_array_size)
readScalarSurfaceType(deltaCoeffs, n_internal_faces, bounfile, "constant", sol, "deltaCoeffs")
print("-"*50+"(deltaCoeffs)")
print(deltaCoeffs.shape)

########################readVectorSurfaceType#############################
Sfx = np.zeros(total_surface_array_size)
Sfy = np.zeros(total_surface_array_size)
Sfz = np.zeros(total_surface_array_size)
readVectorSurfaceType(Sfx,Sfy,Sfz, n_internal_faces, bounfile, "constant/polyMesh", sol, "S")
print("-"*50+"(Sf)")
print(Sfx.shape, Sfy.shape, Sfz.shape)

Cfx = np.zeros(total_surface_array_size)
Cfy = np.zeros(total_surface_array_size)
Cfz = np.zeros(total_surface_array_size)
readVectorSurfaceType(Cfx,Cfy,Cfz, n_internal_faces, bounfile, "constant/polyMesh", sol, "Cf")
print("-"*50+"(Cf)")
print(Cfx.shape, Cfy.shape, Cfz.shape)

deltax = np.zeros(total_surface_array_size)
deltay = np.zeros(total_surface_array_size)
deltaz = np.zeros(total_surface_array_size)
readVectorSurfaceType(deltax,deltay,deltaz, n_internal_faces, bounfile, "constant/polyMesh", sol, "delta")
print("-"*50+"(delta)")
print(deltax.shape, deltay.shape, deltaz.shape)

print("ownersize: ",len(ownerfile.values))
print("neighsize: ",len(neighfile.values))

# time directory dependent field values
# Filter out items that contain digits (this will include floats as strings)
numeric_filter = re.compile(r'^\d*\.?\d+$')
filtered = [item for item in os.listdir(sol) if numeric_filter.match(item)]
# Convert to float and sort in ascending order
sorted_filtered = sorted(filtered, key=float)

print(sorted_filtered)

n_times = len(sorted_filtered)
n_channels = 6 # phi, p_rgh, alpha, Ux, Uy, and Uz
data = np.zeros((n_times, 6, total_surface_array_size))
data_dict = []

for index, time in enumerate(sorted_filtered):
    print(time)
    ########################readScalarSurfaceType#############################
    phi = np.zeros(total_surface_array_size)
    readScalarSurfaceType(phi, n_internal_faces, bounfile, time, sol, "phi")
    print("-"*50+"(phi)")
    print(phi.shape, phi)
    ########################readScalarVolType#############################
    p_rgh = np.zeros(total_surface_array_size)
    readScalarVolType(p_rgh, n_cells, bounfile, time, sol, "p_rgh")
    print("-"*10+"(p_rgh)")
    print(p_rgh.shape, p_rgh)
    alpha = np.zeros(total_surface_array_size)
    readScalarVolType(alpha, n_cells, bounfile, time, sol, "alpha.water")
    print("-"*10+"(alpha.water)")
    print(alpha.shape, alpha)
    ########################readVectorVolType#############################
    Ux = np.zeros(total_surface_array_size)
    Uy = np.zeros(total_surface_array_size)
    Uz = np.zeros(total_surface_array_size)
    readVectorVolType(Ux,Uy,Uz, n_cells, bounfile, time, sol, "U")
    print("-"*50+"(U)")
    print(Ux.shape, Uy.shape, Uz.shape)
    data[index,0] = phi
    data[index,1] = p_rgh
    data[index,2] = alpha
    data[index,3] = Ux
    data[index,4] = Uy
    data[index,5] = Uz

    tmp_dict = {}
    tmp_dict['phi'] = data[index,0] 
    tmp_dict['p_rgh'] = data[index,1] 
    tmp_dict['alpha'] = data[index,2] 
    tmp_dict['Ux'] = data[index,3] 
    tmp_dict['Uy'] = data[index,4] 
    tmp_dict['Uz'] = data[index,5] 
    tmp_dict['time'] = time 

    data_dict.append(tmp_dict)

alphaf = np.zeros(total_surface_array_size)
alpha0 = data_dict[0]['alpha']
alpha1 = data_dict[1]['alpha']
alphaf[n_internal_faces:] = alpha1[n_cells:total_vol_array_size]
phi = data_dict[1]['phi']
phiAlpha = phi*alphaf

prghf = np.zeros(total_surface_array_size)
prgh1 = data_dict[1]['p_rgh']
prghf[n_internal_faces:] = prgh1[n_cells:total_vol_array_size]

rhol = 1.0e3
rhog = 1.0
rhof = np.zeros(total_surface_array_size)
rho1 = alpha1*rhol + (1.0-alpha1)*rhog 
rho0 = alpha0*rhol + (1.0-alpha0)*rhog 
rhof[n_internal_faces:] = rho1[n_cells:total_vol_array_size]

nul = 1.0e-6
nug = 1.48e-5
nuf = np.zeros(total_surface_array_size)
nu1 = alpha1*nul + (1.0-alpha1)*nug 
nuf[n_internal_faces:] = nu1[n_cells:total_vol_array_size]

# examine the vof transport for each cell
for i in range(len(neighfile.values)):
    wn = weights[i]
    wo = 1.0 - wn
    alphaf[i] = wo*alpha1[ownerfile.values[i]] + wn*alpha1[neighfile.values[i]]
    phiAlpha[i] = phi[i]*alphaf[i]

divPhiAlpha = np.zeros(n_cells)
for i in range(len(ownerfile.values)):
    divPhiAlpha[ownerfile.values[i]] += phiAlpha[i]

for i in range(len(neighfile.values)):
    divPhiAlpha[neighfile.values[i]] -= phiAlpha[i]

dAlphadt = ((alpha1 - alpha0)/dt)[:n_cells]
pdeAlpha = dAlphadt + divPhiAlpha/vols

print(pdeAlpha, np.max(pdeAlpha), np.min(pdeAlpha))

# examine the mass conservation for each cell
divPhi = np.zeros(n_cells)
for i in range(len(ownerfile.values)):
    divPhi[ownerfile.values[i]] += phi[i]

for i in range(len(neighfile.values)):
    divPhi[neighfile.values[i]] -= phi[i]

print(divPhi/vols, np.max(divPhi/vols), np.min(divPhi/vols))

# examine the pEqn for the given timestep 
# grad(U)

Ux0 = data_dict[0]['Ux']
Uy0 = data_dict[0]['Uy']
Uz0 = data_dict[0]['Uz']

Ux1 = data_dict[1]['Ux']
Uy1 = data_dict[1]['Uy']
Uz1 = data_dict[1]['Uz']

Ufx = np.zeros(total_surface_array_size)
Ufy = np.zeros(total_surface_array_size)
Ufz = np.zeros(total_surface_array_size)

Ufx[n_internal_faces:] = Ux1[n_cells:total_vol_array_size]
Ufy[n_internal_faces:] = Uy1[n_cells:total_vol_array_size]
Ufz[n_internal_faces:] = Uz1[n_cells:total_vol_array_size]

# surface interpolation (U,prgh)
for facei in range(len(neighfile.values)):
    celli_o = ownerfile.values[facei]
    celli_n = neighfile.values[facei]
    wn = weights[facei]
    wo = 1.0 - wn
    Ufx[facei] = wo*Ux1[celli_o] + wn*Ux1[celli_n]
    Ufy[facei] = wo*Uy1[celli_o] + wn*Uy1[celli_n]
    Ufz[facei] = wo*Uz1[celli_o] + wn*Uz1[celli_n]
    prghf[facei] = wo*prgh1[celli_o] + wn*prgh1[celli_n]
    rhof[facei] = wo*rho1[celli_o] + wn*rho1[celli_n]
    nuf[facei] = wo*nu1[celli_o] + wn*nu1[celli_n]

# grad(U)
gradUxx = np.zeros(total_vol_array_size)
gradUxy = np.zeros(total_vol_array_size)
gradUxz = np.zeros(total_vol_array_size)

gradUyx = np.zeros(total_vol_array_size)
gradUyy = np.zeros(total_vol_array_size)
gradUyz = np.zeros(total_vol_array_size)

gradUzx = np.zeros(total_vol_array_size)
gradUzy = np.zeros(total_vol_array_size)
gradUzz = np.zeros(total_vol_array_size)

Sfx_Ufx = Sfx*Ufx
Sfx_Ufy = Sfx*Ufy
Sfx_Ufz = Sfx*Ufz

Sfy_Ufx = Sfy*Ufx
Sfy_Ufy = Sfy*Ufy
Sfy_Ufz = Sfy*Ufz

Sfz_Ufx = Sfz*Ufx
Sfz_Ufy = Sfz*Ufy
Sfz_Ufz = Sfz*Ufz

# grad(alpha)
gradAlphax = np.zeros(total_vol_array_size)
gradAlphay = np.zeros(total_vol_array_size)
gradAlphaz = np.zeros(total_vol_array_size)
alphaf_Sfx = alphaf*Sfx
alphaf_Sfy = alphaf*Sfy
alphaf_Sfz = alphaf*Sfz

# grad(p_rgh)
gradPrghx = np.zeros(total_vol_array_size)
gradPrghy = np.zeros(total_vol_array_size)
gradPrghz = np.zeros(total_vol_array_size)
prghf_Sfx = prghf*Sfx
prghf_Sfy = prghf*Sfy
prghf_Sfz = prghf*Sfz

# grad(rho)
gradRhox = np.zeros(total_vol_array_size)
gradRhoy = np.zeros(total_vol_array_size)
gradRhoz = np.zeros(total_vol_array_size)
rho_Sfx = rhof*Sfx
rho_Sfy = rhof*Sfy
rho_Sfz = rhof*Sfz

for facei in range(len(ownerfile.values)):
    celli_o = ownerfile.values[facei]
    gradUxx[celli_o] += Sfx_Ufx[facei]
    gradUxy[celli_o] += Sfx_Ufy[facei]
    gradUxz[celli_o] += Sfx_Ufz[facei]

    gradUyx[celli_o] += Sfy_Ufx[facei]
    gradUyy[celli_o] += Sfy_Ufy[facei]
    gradUyz[celli_o] += Sfy_Ufz[facei]

    gradUzx[celli_o] += Sfz_Ufx[facei]
    gradUzy[celli_o] += Sfz_Ufy[facei]
    gradUzz[celli_o] += Sfz_Ufz[facei]

    gradAlphax[celli_o] += alphaf_Sfx[facei]
    gradAlphay[celli_o] += alphaf_Sfy[facei]
    gradAlphaz[celli_o] += alphaf_Sfz[facei]

    gradPrghx[celli_o] += prghf_Sfx[facei]
    gradPrghy[celli_o] += prghf_Sfy[facei]
    gradPrghz[celli_o] += prghf_Sfz[facei]

    gradRhox[celli_o] += rho_Sfx[facei]
    gradRhoy[celli_o] += rho_Sfy[facei]
    gradRhoz[celli_o] += rho_Sfz[facei]

for facei in range(len(neighfile.values)):
    celli_n = neighfile.values[facei]
    gradUxx[celli_n] -= Sfx_Ufx[facei]
    gradUxy[celli_n] -= Sfx_Ufy[facei]
    gradUxz[celli_n] -= Sfx_Ufz[facei]

    gradUyx[celli_n] -= Sfy_Ufx[facei]
    gradUyy[celli_n] -= Sfy_Ufy[facei]
    gradUyz[celli_n] -= Sfy_Ufz[facei]

    gradUzx[celli_n] -= Sfz_Ufx[facei]
    gradUzy[celli_n] -= Sfz_Ufy[facei]
    gradUzz[celli_n] -= Sfz_Ufz[facei]

    gradAlphax[celli_n] -= alphaf_Sfx[facei]
    gradAlphay[celli_n] -= alphaf_Sfy[facei]
    gradAlphaz[celli_n] -= alphaf_Sfz[facei]

    gradPrghx[celli_n] -= prghf_Sfx[facei]
    gradPrghy[celli_n] -= prghf_Sfy[facei]
    gradPrghz[celli_n] -= prghf_Sfz[facei]

    gradRhox[celli_n] -= rho_Sfx[facei]
    gradRhoy[celli_n] -= rho_Sfy[facei]
    gradRhoz[celli_n] -= rho_Sfz[facei]

gradUxx[:n_cells] = gradUxx[:n_cells]/vols
gradUxy[:n_cells] = gradUxy[:n_cells]/vols
gradUxz[:n_cells] = gradUxz[:n_cells]/vols

gradUyx[:n_cells] = gradUyx[:n_cells]/vols
gradUyy[:n_cells] = gradUyy[:n_cells]/vols
gradUyz[:n_cells] = gradUyz[:n_cells]/vols

gradUzx[:n_cells] = gradUzx[:n_cells]/vols
gradUzy[:n_cells] = gradUzy[:n_cells]/vols
gradUzz[:n_cells] = gradUzz[:n_cells]/vols

gradAlphax[:n_cells] = gradAlphax[:n_cells]/vols
gradAlphay[:n_cells] = gradAlphay[:n_cells]/vols
gradAlphaz[:n_cells] = gradAlphaz[:n_cells]/vols

gradPrghx[:n_cells] = gradPrghx[:n_cells]/vols
gradPrghy[:n_cells] = gradPrghy[:n_cells]/vols
gradPrghz[:n_cells] = gradPrghz[:n_cells]/vols

gradRhox[:n_cells] = gradRhox[:n_cells]/vols
gradRhoy[:n_cells] = gradRhoy[:n_cells]/vols
gradRhoz[:n_cells] = gradRhoz[:n_cells]/vols

# surface interpolation (grad(U),grad(prgh),grad(alpha),grad(rho))
gradUxxf = np.zeros(total_surface_array_size)
gradUxyf = np.zeros(total_surface_array_size)
gradUxzf = np.zeros(total_surface_array_size)

gradUyxf = np.zeros(total_surface_array_size)
gradUyyf = np.zeros(total_surface_array_size)
gradUyzf = np.zeros(total_surface_array_size)

gradUzxf = np.zeros(total_surface_array_size)
gradUzyf = np.zeros(total_surface_array_size)
gradUzzf = np.zeros(total_surface_array_size)

gradAlphaxf = np.zeros(total_surface_array_size)
gradAlphayf = np.zeros(total_surface_array_size)
gradAlphazf = np.zeros(total_surface_array_size)

gradPrghxf = np.zeros(total_surface_array_size)
gradPrghyf = np.zeros(total_surface_array_size)
gradPrghzf = np.zeros(total_surface_array_size)

gradRhoxf = np.zeros(total_surface_array_size)
gradRhoyf = np.zeros(total_surface_array_size)
gradRhozf = np.zeros(total_surface_array_size)

# looping over internal faces
for i in range(len(neighfile.values)):
    celli_o = ownerfile.values[i]
    celli_n = neighfile.values[i]
    wn = weights[i]
    wo = 1.0 - wn
    gradUxxf[i] = wo*gradUxx[celli_o] + wn*gradUxx[celli_n]
    gradUxyf[i] = wo*gradUxy[celli_o] + wn*gradUxy[celli_n]
    gradUxzf[i] = wo*gradUxz[celli_o] + wn*gradUxz[celli_n]

    gradUyxf[i] = wo*gradUyx[celli_o] + wn*gradUyx[celli_n]
    gradUyyf[i] = wo*gradUyy[celli_o] + wn*gradUyy[celli_n]
    gradUyzf[i] = wo*gradUyz[celli_o] + wn*gradUyz[celli_n]

    gradUzxf[i] = wo*gradUzx[celli_o] + wn*gradUzx[celli_n]
    gradUzyf[i] = wo*gradUzy[celli_o] + wn*gradUzy[celli_n]
    gradUzzf[i] = wo*gradUzz[celli_o] + wn*gradUzz[celli_n]

    gradAlphaxf[i] = wo*gradAlphax[celli_o] + wn*gradAlphax[celli_n]
    gradAlphayf[i] = wo*gradAlphay[celli_o] + wn*gradAlphay[celli_n]
    gradAlphazf[i] = wo*gradAlphaz[celli_o] + wn*gradAlphaz[celli_n]

    gradPrghxf[i] = wo*gradPrghx[celli_o] + wn*gradPrghx[celli_n]
    gradPrghyf[i] = wo*gradPrghy[celli_o] + wn*gradPrghy[celli_n]
    gradPrghzf[i] = wo*gradPrghz[celli_o] + wn*gradPrghz[celli_n]

    gradRhoxf[i] = wo*gradRhox[celli_o] + wn*gradRhox[celli_n]
    gradRhoyf[i] = wo*gradRhoy[celli_o] + wn*gradRhoy[celli_n]
    gradRhozf[i] = wo*gradRhoz[celli_o] + wn*gradRhoz[celli_n]

# loop over faces at boundaries
nx = Sfx/(magSf+1.0e-16)
ny = Sfy/(magSf+1.0e-16)
nz = Sfz/(magSf+1.0e-16)
for facei in range(len(neighfile.values), len(ownerfile.values),1):
    celli_o = ownerfile.values[facei]

    deltafO = deltaCoeffs[facei]
    correctU_x = deltafO*(Ufx[facei] - Ux1[celli_o]) - (nx[facei]*gradUxx[celli_o]+ny[facei]*gradUyx[celli_o]+nz[facei]*gradUzx[celli_o])
    correctU_y = deltafO*(Ufy[facei] - Uy1[celli_o]) - (nx[facei]*gradUxy[celli_o]+ny[facei]*gradUyy[celli_o]+nz[facei]*gradUzy[celli_o])
    correctU_z = deltafO*(Ufz[facei] - Uz1[celli_o]) - (nx[facei]*gradUxz[celli_o]+ny[facei]*gradUyz[celli_o]+nz[facei]*gradUzz[celli_o])

    correctAlpha = deltafO*(alphaf[facei] - alpha1[celli_o]) - (nx[facei]*gradAlphax[celli_o]+ny[facei]*gradAlphay[celli_o]+nz[facei]*gradAlphaz[celli_o])

    correctPrgh = deltafO*(prghf[facei] - prgh1[celli_o]) - (nx[facei]*gradPrghx[celli_o]+ny[facei]*gradPrghy[celli_o]+nz[facei]*gradPrghz[celli_o])

    correctRho = deltafO*(rhof[facei] - rho1[celli_o]) - (nx[facei]*gradRhox[celli_o]+ny[facei]*gradRhoy[celli_o]+nz[facei]*gradRhoz[celli_o])

    gradUxxf[facei] = gradUxx[celli_o] + nx[facei]*correctU_x
    gradUxyf[facei] = gradUxy[celli_o] + nx[facei]*correctU_y
    gradUxzf[facei] = gradUxz[celli_o] + nx[facei]*correctU_z

    gradUyxf[facei] = gradUyx[celli_o] + ny[facei]*correctU_x
    gradUyyf[facei] = gradUyy[celli_o] + ny[facei]*correctU_y
    gradUyzf[facei] = gradUyz[celli_o] + ny[facei]*correctU_z

    gradUzxf[facei] = gradUzx[celli_o] + nz[facei]*correctU_x
    gradUzyf[facei] = gradUzy[celli_o] + nz[facei]*correctU_y
    gradUzzf[facei] = gradUzz[celli_o] + nz[facei]*correctU_z

    gradAlphaxf[facei] = gradAlphax[celli_o] + nx[facei]*correctAlpha
    gradAlphayf[facei] = gradAlphay[celli_o] + ny[facei]*correctAlpha
    gradAlphazf[facei] = gradAlphaz[celli_o] + nz[facei]*correctAlpha

    gradPrghxf[facei] = gradPrghx[celli_o] + nx[facei]*correctPrgh
    gradPrghyf[facei] = gradPrghy[celli_o] + ny[facei]*correctPrgh
    gradPrghzf[facei] = gradPrghz[celli_o] + nz[facei]*correctPrgh

    gradRhoxf[facei] = gradRhox[celli_o] + nx[facei]*correctRho
    gradRhoyf[facei] = gradRhoy[celli_o] + ny[facei]*correctRho
    gradRhozf[facei] = gradRhoz[celli_o] + nz[facei]*correctRho

gradUxx[n_cells:total_vol_array_size] = gradUxxf[n_internal_faces:total_surface_array_size]
gradUxy[n_cells:total_vol_array_size] = gradUxyf[n_internal_faces:total_surface_array_size]
gradUxz[n_cells:total_vol_array_size] = gradUxzf[n_internal_faces:total_surface_array_size]

gradUyx[n_cells:total_vol_array_size] = gradUyxf[n_internal_faces:total_surface_array_size]
gradUyy[n_cells:total_vol_array_size] = gradUyyf[n_internal_faces:total_surface_array_size]
gradUyz[n_cells:total_vol_array_size] = gradUyzf[n_internal_faces:total_surface_array_size]

gradUzx[n_cells:total_vol_array_size] = gradUzxf[n_internal_faces:total_surface_array_size]
gradUzy[n_cells:total_vol_array_size] = gradUzyf[n_internal_faces:total_surface_array_size]
gradUzz[n_cells:total_vol_array_size] = gradUzzf[n_internal_faces:total_surface_array_size]

gradAlphax[n_cells:total_vol_array_size] = gradAlphaxf[n_internal_faces:total_surface_array_size]
gradAlphay[n_cells:total_vol_array_size] = gradAlphayf[n_internal_faces:total_surface_array_size]
gradAlphaz[n_cells:total_vol_array_size] = gradAlphazf[n_internal_faces:total_surface_array_size]

gradRhox[n_cells:total_vol_array_size] = gradRhoxf[n_internal_faces:total_surface_array_size]
gradRhoy[n_cells:total_vol_array_size] = gradRhoyf[n_internal_faces:total_surface_array_size]
gradRhoz[n_cells:total_vol_array_size] = gradRhozf[n_internal_faces:total_surface_array_size]

gradPrghx[n_cells:total_vol_array_size] = gradPrghxf[n_internal_faces:total_surface_array_size]
gradPrghy[n_cells:total_vol_array_size] = gradPrghyf[n_internal_faces:total_surface_array_size]
gradPrghz[n_cells:total_vol_array_size] = gradPrghzf[n_internal_faces:total_surface_array_size]

magGradU = np.sqrt(gradUxx*gradUxx + gradUxy*gradUxy + gradUxz*gradUxz 
                   + gradUyx*gradUyx + gradUyy*gradUyy + gradUyz*gradUyz  
                   + gradUzx*gradUzx + gradUzy*gradUzy + gradUzz*gradUzz)

magGradAlphaf = np.sqrt(gradAlphaxf*gradAlphaxf+gradAlphayf*gradAlphayf+gradAlphazf*gradAlphazf)
n_alpha_x = gradAlphaxf/(magGradAlphaf+1.0e-16)
n_alpha_y = gradAlphayf/(magGradAlphaf+1.0e-16)
n_alpha_z = gradAlphazf/(magGradAlphaf+1.0e-16)

kappa = np.zeros(total_vol_array_size)

for facei in range(len(ownerfile.values)):
    celli_o = ownerfile.values[facei]
    kappa[celli_o] += n_alpha_x[facei]*Sfx[facei] + n_alpha_y[facei]*Sfy[facei] + n_alpha_z[facei]*Sfz[facei]

    if facei < n_internal_faces:
        celli_n = neighfile.values[facei]
        kappa[celli_n] -= n_alpha_x[facei]*Sfx[facei] + n_alpha_y[facei]*Sfy[facei] + n_alpha_z[facei]*Sfz[facei]
kappa[:n_cells] = kappa[:n_cells]/vols

# apply extrapolateCalculated boundary condition
for facei in range(len(neighfile.values),len(ownerfile.values),1):
    kappa[n_cells+facei-n_internal_faces] = kappa[ownerfile.values[facei]]

#print(f"max/min:{max(magGradU)} {min(magGradU)}")
#print(f"max/min:{max(gradUxx)} {min(gradUxx)}")
#print(f"max/min:{max(gradUxy)} {min(gradUxy)}")
#print(f"max/min:{max(gradUxz)} {min(gradUxz)}")
#print(f"max/min:{max(gradUyx)} {min(gradUyx)}")
#print(f"max/min:{max(gradUyy)} {min(gradUyy)}")
#print(f"max/min:{max(gradUyz)} {min(gradUyz)}")
#print(f"max/min:{max(gradUzx)} {min(gradUzx)}")
#print(f"max/min:{max(gradUzy)} {min(gradUzy)}")
#print(f"max/min:{max(gradUzz)} {min(gradUzz)}")
#print(f"(weights)max/min:{max(weights[:n_internal_faces])} {min(weights[:n_internal_faces])}")

# calculate A (boundary values are zero)
A = np.zeros(total_vol_array_size)
for facei in range(len(ownerfile.values)):
    celli_o = ownerfile.values[facei]
    wn = weights[facei]
    wo = 1.0 - wn
    delta = deltaCoeffs[facei]
    A[celli_o] += rhof[facei]*phi[facei]*wo
    A[celli_o] += rhof[facei]*nu1[facei]*delta*magSf[facei]

    if facei < n_internal_faces:
        celli_n = neighfile.values[facei]
        A[celli_n] -= rhof[facei]*phi[facei]*wn
        A[celli_n] += rhof[facei]*nu1[facei]*delta*magSf[facei]

A[:n_cells] = A[:n_cells]/vols + rho1[:n_cells]/dt

# calculate H (boundary values are zero)
Hx = np.zeros(total_vol_array_size)
Hy = np.zeros(total_vol_array_size)
Hz = np.zeros(total_vol_array_size)
for facei in range(len(ownerfile.values)):
    celli_o = ownerfile.values[facei]
    divU = gradUxxf[facei] + gradUyyf[facei] + gradUzzf[facei]
    dev2_gradU_t_xx = gradUxxf[facei] - 2./3.0*divU
    dev2_gradU_t_xy = gradUyxf[facei]
    dev2_gradU_t_xz = gradUzxf[facei]

    dev2_gradU_t_yx = gradUxyf[facei]
    dev2_gradU_t_yy = gradUyyf[facei] - 2./3.0*divU
    dev2_gradU_t_yz = gradUzyf[facei]

    dev2_gradU_t_zx = gradUxzf[facei]
    dev2_gradU_t_zy = gradUyzf[facei]
    dev2_gradU_t_zz = gradUzzf[facei] - 2./3.0*divU

    if facei < n_internal_faces:
        wn = weights[facei]
        wo = 1.0 - wn
        delta = deltaCoeffs[facei]
        # convection (owner)
        Hx[celli_o] -= rhof[facei]*phi[facei]*wn*Ux1[celli_n]
        Hy[celli_o] -= rhof[facei]*phi[facei]*wn*Uy1[celli_n]
        Hz[celli_o] -= rhof[facei]*phi[facei]*wn*Uz1[celli_n]
        # conduction (owner)
        Hx[celli_o] += rhof[facei]*nu1[facei]*delta*magSf[facei]*Ux1[celli_n]
        Hy[celli_o] += rhof[facei]*nu1[facei]*delta*magSf[facei]*Uy1[celli_n]
        Hz[celli_o] += rhof[facei]*nu1[facei]*delta*magSf[facei]*Uz1[celli_n]
        celli_n = neighfile.values[facei]
        # dev2(owner)
        Hx[celli_o] -= rhof[facei]*nu1[facei]*(Sfx[facei]*dev2_gradU_t_xx + Sfy[facei]*dev2_gradU_t_yx + Sfz[facei]*dev2_gradU_t_zx)
        Hy[celli_o] -= rhof[facei]*nu1[facei]*(Sfx[facei]*dev2_gradU_t_xy + Sfy[facei]*dev2_gradU_t_yy + Sfz[facei]*dev2_gradU_t_zy)
        Hz[celli_o] -= rhof[facei]*nu1[facei]*(Sfx[facei]*dev2_gradU_t_xz + Sfy[facei]*dev2_gradU_t_yz + Sfz[facei]*dev2_gradU_t_zz)

        # convection (neigh)
        Hx[celli_n] += rhof[facei]*phi[facei]*wo*Ux1[celli_o]
        Hy[celli_n] += rhof[facei]*phi[facei]*wo*Uy1[celli_o]
        Hz[celli_n] += rhof[facei]*phi[facei]*wo*Uz1[celli_o]
        # conduction (neigh)
        Hx[celli_n] += rhof[facei]*nu1[facei]*delta*magSf[facei]*Ux1[celli_o]
        Hy[celli_n] += rhof[facei]*nu1[facei]*delta*magSf[facei]*Uy1[celli_o]
        Hz[celli_n] += rhof[facei]*nu1[facei]*delta*magSf[facei]*Uz1[celli_o]
        # dev2(neigh)
        Hx[celli_n] += rhof[facei]*nu1[facei]*(Sfx[facei]*dev2_gradU_t_xx + Sfy[facei]*dev2_gradU_t_yx + Sfz[facei]*dev2_gradU_t_zx)
        Hy[celli_n] += rhof[facei]*nu1[facei]*(Sfx[facei]*dev2_gradU_t_xy + Sfy[facei]*dev2_gradU_t_yy + Sfz[facei]*dev2_gradU_t_zy)
        Hz[celli_n] += rhof[facei]*nu1[facei]*(Sfx[facei]*dev2_gradU_t_xz + Sfy[facei]*dev2_gradU_t_yz + Sfz[facei]*dev2_gradU_t_zz)

    else:
        # account for the boundary contribution (take values from boundary)
        # convection
        Hx[celli_o] -= rhof[facei]*phi[facei]*Ufx[facei]
        Hy[celli_o] -= rhof[facei]*phi[facei]*Ufy[facei]
        Hz[celli_o] -= rhof[facei]*phi[facei]*Ufz[facei]
        # conduction
        Hx[celli_o] += rhof[facei]*nu1[facei]*(Sfx[facei]*gradUxxf[facei]+Sfy[facei]*gradUyxf[facei]+Sfz[facei]*gradUzxf[facei])
        Hy[celli_o] += rhof[facei]*nu1[facei]*(Sfx[facei]*gradUxyf[facei]+Sfy[facei]*gradUyyf[facei]+Sfz[facei]*gradUzyf[facei])
        Hz[celli_o] += rhof[facei]*nu1[facei]*(Sfx[facei]*gradUxzf[facei]+Sfy[facei]*gradUyzf[facei]+Sfz[facei]*gradUzzf[facei])
        # dev2
        Hx[celli_o] += rhof[facei]*nu1[facei]*(Sfx[facei]*dev2_gradU_t_xx + Sfy[facei]*dev2_gradU_t_yx + Sfz[facei]*dev2_gradU_t_zx)
        Hy[celli_o] += rhof[facei]*nu1[facei]*(Sfx[facei]*dev2_gradU_t_xy + Sfy[facei]*dev2_gradU_t_yy + Sfz[facei]*dev2_gradU_t_zy)
        Hz[celli_o] += rhof[facei]*nu1[facei]*(Sfx[facei]*dev2_gradU_t_xz + Sfy[facei]*dev2_gradU_t_yz + Sfz[facei]*dev2_gradU_t_zz)

Hx[:n_cells] = Hx[:n_cells]/vols + rho0[:n_cells]*Ux0[:n_cells]/dt
Hy[:n_cells] = Hy[:n_cells]/vols + rho0[:n_cells]*Uy0[:n_cells]/dt
Hz[:n_cells] = Hz[:n_cells]/vols + rho0[:n_cells]*Uz0[:n_cells]/dt

time = data_dict[1]["time"]
writeScalarVolType(A,n_cells,bounfile,time,sol,"UEqnA")
writeVectorVolType(Hx,Hy,Hz,n_cells,bounfile,time,sol,"UEqnH")
writeScalarVolType(magGradU,n_cells,bounfile,time,sol,"magGradU")
writeVectorVolType(gradAlphax,gradAlphay,gradAlphaz,n_cells,bounfile,time,sol,"gradAlpha")
writeVectorVolType(gradPrghx,gradPrghy,gradPrghz,n_cells,bounfile,time,sol,"gradPrgh")
writeVectorVolType(gradRhox,gradRhoy,gradRhoz,n_cells,bounfile,time,sol,"gradRho")
writeVectorVolType(Ux1,Uy1,Uz1,n_cells,bounfile,time,sol,"U")
writeTensorVolType(gradUxx,gradUxy,gradUxz,gradUyx,gradUyy,gradUyz,gradUzx,gradUzy,gradUzz,n_cells,bounfile,time,sol,"gradU")

writeScalarSurfaceType(alphaf,n_internal_faces,bounfile,time,sol,"alphaf")
writeScalarSurfaceType(magSf,n_internal_faces,bounfile,time,sol,"magSf")
writeVectorSurfaceType(Cfx,Cfy,Cfz,n_internal_faces,bounfile,time,sol,"Cf")
writeVectorSurfaceType(Ufx,Ufy,Ufz,n_internal_faces,bounfile,time,sol,"Uf")
writeVectorSurfaceType(Sfx,Sfy,Sfz,n_internal_faces,bounfile,time,sol,"Sf")
