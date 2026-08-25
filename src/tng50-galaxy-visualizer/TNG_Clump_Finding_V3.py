# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 19:05:03 2024

@author: josit
"""
# This code will only be making plots for the young stars and cool gas

import os
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
import numpy as np
import scipy.constants as const
import illustris_python as il
import sys, time
import sklearn.neighbors as nb
from shutil import rmtree
import matplotlib.pyplot as plt
import multiprocessing
import gc
import colourhist as co
import yt
from yt.data_objects.level_sets.api import *
from yt.data_objects.level_sets.clump_validators import add_validator

##########################################################################################

def W(h, d):
    sig = 1/(np.pi * h ** 3)
    # d and h need to be in the same units
    q = d/h
    
    condition_1 = np.logical_and(0 <= q, q <= 1)
    condition_2 = np.logical_and(1 < q, q <= 2) #default values are 1 & 2
    
    result = np.zeros_like(q)
    
    result[condition_1] = sig * (1 - 1.5 * (q[condition_1] ** 2) * (1 - q[condition_1]/2))
    result[condition_2] = (sig / 4) * (2 - q[condition_2]) ** 3
    
    return (result)

def processing(function, threads, args):
    pool = multiprocessing.Pool(threads)
    out = pool.map(function, args)  
    pool.close()
    pool.join()
    del pool
    return (out)

def flatten_clumps(clump):
    clumps = [clump]
    for child in clump.children:
        clumps.extend(flatten_clumps(child))
    return clumps

def get_leaf_clumps(clump):
    if len(clump.children) == 0:  # no further refinement
        return [clump]
    leaves = []
    for child in clump.children:
        leaves.extend(get_leaf_clumps(child))
    return leaves
def rotation_matrix_from_vectors(a, b):
    """
    Return rotation matrix R (3x3) that rotates vector a to vector b.
    Uses robust formula including 180-degree case.
    a, b : length-3 arrays (need not be unit length).
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.allclose(c, 1.0):
        return np.eye(3)   # already aligned
    if np.allclose(c, -1.0):
        # 180-degree rotation: need an arbitrary perpendicular axis
        # choose axis orthogonal to a
        axis = np.array([1.0, 0.0, 0.0])
        if np.allclose(np.abs(a), axis):
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis)
        v = v / np.linalg.norm(v)
        return -np.eye(3) + 2.0 * np.outer(v, v)
    s = np.linalg.norm(v)
    kmat = np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],     0]])
    R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))
    return R    

##########################################################################################


#FileNumber = 25
ID = int(sys.argv[1])
FileNumber = int(sys.argv[2])
z_redshift = float(sys.argv[3])
Prefix = 'TNG50'

print('This code is being ran')

basePath = '/home/malcolmp/projects/rrg-sellison/Illustris-TNG/SimData/TNG50-1/output'

#IDs = [145922, 146923, 149244, 156279, 157767, 160186, 162973, 164892, 165799, 168663] # For snapshot 33
# IDs = [3769,3770, 16024,17197,17198,18517,21550,21551,23047,23048,23050,24656]

particleType = ['gas', 'stars','dm']
propertiesDM = ['Velocities', 'Coordinates', 'SubfindHsml', 'SubfindDMDensity', 'ParticleIDs']
propertiesStars = ['Masses', 'Velocities', 'Coordinates', 'StellarHsml', 'GFM_StellarFormationTime', 'ParticleIDs', 'GFM_InitialMass', 'GFM_MetalsTagged']
propertiesGas = ['Masses', 'Velocities', 'Coordinates', 'Density', 'SubfindHsml', 'InternalEnergy', 'GFM_MetalsTagged', 'ElectronAbundance']

#z_redshift = 3 # JW: renamed z (z is used as a coordinate later)
a = 1 / (1 + z_redshift)
h_cosmo = cosmo.H(0).value / 100
nn_multiplier = 3 # default value is 3, this multiplier increases the radius of boxes that is used around a single raw data point


home = "/home/malcolmp/"
#codedir=f'{home}scratch/{Prefix}/Snapshot_{FileNumber:03}/TNG_{ID}/'
#datadir=f'{home}scratch/{Prefix}/Snapshot_{FileNumber:03}/TMG_{ID}/'
plotsdir=f'{home}scratch/Plots/Clump_Finder_Plots/{Prefix}/Snapshot_{FileNumber:03}/TNG_{ID}/'
clumpsdir=f'{plotsdir}/Clump_Files/'

os.makedirs(plotsdir, exist_ok = True)
os.makedirs(clumpsdir, exist_ok = True)


do_dm = True

clump_dm_limit = 100 # number density
clump_dens_min = 100 # young stellar number density

cmap = 'nipy_spectral'

##########################################################################################

SubHaloGas = il.snapshot.loadSubhalo(basePath, FileNumber, id=ID, partType = particleType[0], fields=propertiesGas)

SubHalo = il.snapshot.loadSingle(basePath, FileNumber, subhaloID=ID)

Origin = SubHalo['SubhaloPos'] * a / h_cosmo
GrNr = SubHalo['SubhaloGrNr']

Halo = il.groupcat.loadSingle(basePath, FileNumber, haloID = GrNr)
Rvir4 = str(4*Halo['Group_R_Crit200'] * a / h_cosmo)[:4]

SubHaloStars = il.snapshot.loadSubhalo(basePath, FileNumber, id=ID, partType = particleType[1], fields=propertiesStars)

# plt.figure()
# plt.scatter(star_coords[:,0], star_coords[:,1])
# plt.savefig(f'/home/malcolmp/scratch/temp/{ID}.png')
# continue

##################################################################################################################################################

HalfMassRadius = SubHalo['SubhaloHalfmassRadType'][0] * a / h_cosmo
#HalfMassRadius3 = 3*np.std(star_coords)
HalfMassRadius3 = 14
print(f'3x Half Mass Radius = {HalfMassRadius3}')
# We now put the coordinates into units of kpc from ckpc/h
edited_coordinates = SubHaloGas['Coordinates'] * a / h_cosmo - Origin  # We do this here a first time in order to cut out unwanted data
# JW: some things to determine the best grid size
# Use gas half-mass radius instead or use virial radius (maybe)
mask_goodgas = ((np.abs(edited_coordinates[:,0]) < HalfMassRadius3) & (np.abs(edited_coordinates[:,1]) < HalfMassRadius3) &
                (np.abs(edited_coordinates[:,2]) < HalfMassRadius3)) #np.std(star_coords)


for key in SubHaloGas.keys():
    if key != 'count':
        if key not in ['Velocities', 'Coordinates']: SubHaloGas[key] = SubHaloGas[key][mask_goodgas]
        else: SubHaloGas[key] = SubHaloGas[key][mask_goodgas, :]

# We need to do is calculate our value of h so we can convert any and all units as necessary
# Here is where we calculate our h value (the radius of our density sphere to be used by W)

Vol = SubHaloGas['Masses']/SubHaloGas['Density']
# This h is for smoothing, not the Hubble Constant for coordinates
factor = 1.0 # JW: Dylan Nelson told me this should be 2.5, but I found that factor=1.0 makes the plot on the bottom
             # more closely resembles the visualization API. Not sure why
h_gas = factor * ((0.75 * Vol/ np.pi) ** (1/3)) #JW: factor
h_gas *= a / h_cosmo # Converts the length from comoving kpc/h to physical kpc

##################################################################################################################################################

# Convert any properties that need to be converted to correct units

# Temperature
InternalEnergy = SubHaloGas['InternalEnergy'] * 1e6 # m^2/s^2
xe = SubHaloGas['ElectronAbundance']
Xh = 0.76
gamma = 5/3
kb = const.k #kb in J/K
mu = 4 * const.m_p / (1 + 3 * Xh + 4 * Xh * xe) # units of kg
T = (gamma - 1) * (InternalEnergy/kb) * mu

plt.figure()
#plt.hist(np.log10(T), bins = 50)
#plt.savefig(f'/home/malcolmp/scratch/Plots/Temp/Histogram_{ID}.png')

#continue

##################################################################################################################################################

v_x = SubHaloGas['Velocities'][:,0] * (np.sqrt(a))
v_y = SubHaloGas['Velocities'][:,1] * (np.sqrt(a))
v_z = SubHaloGas['Velocities'][:,2] * (np.sqrt(a))

##################################################################################################################################################

edited_coordinates = SubHaloGas['Coordinates'] * a / h_cosmo - Origin  #JW: now centered on subhalo centre

min_all,max_all = -HalfMassRadius3, HalfMassRadius3#np.min(edited_coordinates),np.max(edited_coordinates) # the min and max coordinate in x,y or z
maxdelta = 2 * HalfMassRadius3 #max_all-min_all; These should technically be the same, but this is more obvious

best_resolution = 0.10 #np.min([np.min(h_gas),0.07]) For now we want everything to be 140pc
print('Max delta: ', maxdelta, "\nBest Resolution: ",best_resolution)

n = int(np.ceil(maxdelta / best_resolution)) #JW: Nir says to make the grid cells to be twice the best resolution, (in IllustrisTNG the best resolution is 140pc)

#n = np.min([n,400])  # We are using 400 for the time being, though 500 should be checked
#n = 500
print('Grid Dimensions: ', n)


# Make a raw data array that will contain all of the raw data for different particle data types
raw_data = np.zeros([11,len(T)])
raw_data[0] = T
raw_data[1] = v_x
raw_data[2] = v_y
raw_data[3] = v_z
raw_data[4] = SubHaloGas['GFM_MetalsTagged'][:,0]
raw_data[5] = SubHaloGas['GFM_MetalsTagged'][:,1]
raw_data[10] = SubHaloGas['Masses'] / h_cosmo  # 10^10 Msun

raw_density = SubHaloGas['Density'] * h_cosmo**2 / a ** 3 # 10^10 Msun/kpc^3
#mask = (np.abs(edited_coordinates[:,0]) < 10) & (np.abs(edited_coordinates[:,1]) < 10) & (np.abs(edited_coordinates[:,2]) < 10); I believe this currently has no function


#######################################################################################################################
# Vector time
#######################################################################################################################
# Before we make the 3D cubic grid, we need to rotate all of the coordinates so that we look at the galaxy face-on
# Stack position and velocity components into (N, 3) arrays
r = edited_coordinates       # shape (N, 3)
v = np.column_stack((raw_data[1], raw_data[2], raw_data[3])) # shape (N, 3)

# Cross product for each particle
L = np.cross(r, v)   # shape (N, 3)
L_total = np.sum(L * raw_data[10][:, None], axis = 0)
L_hat = L_total / np.linalg.norm(L_total)

z_axis = np.array([0.0, 0.0, 1.0])
R = rotation_matrix_from_vectors(L_hat, z_axis)

r_rot = r @ R.T
v_rot = v @ R.T
print(L_hat)

L = np.cross(r, v)   # shape (N, 3)
L_total = np.sum(L * raw_data[10][:, None], axis = 0) @ R.T
L_hat = L_total / np.linalg.norm(L_total)
print(L_hat)
#print(r_rot[0:10])
#print(n * r_rot[0:10])
#print((n * r_rot[0:10]).astype('int'))

edited_coordinates = edited_coordinates @ R.T
raw_data[1] = v[:,0]
raw_data[2] = v[:,1]
raw_data[3] = v[:,2]


##################################################################################################################################################


# Now we need to fit the coordinates to our grid using the nearest neighbors

grid_shape = (n, n, n)
# dx is the length of one side of a cell
dx = maxdelta/n #JW

# Coordinates
coord_range = np.linspace(min_all,max_all,n).astype('float32') #JW: make all directions go from min_all to max_all
#y_range = np.linspace(min_all,max_all,n).astype('float32') #JW
#z_range = np.linspace(min_all,max_all,n).astype('float32') #JW

# Indices of Coordinates
coord_index_range = (np.linspace(0, grid_shape[0]-1, num=grid_shape[0])).astype(int)
#y_index_range = (np.linspace(0, grid_shape[0]-1, num=grid_shape[0])).astype(int)
#z_index_range = (np.linspace(0, grid_shape[0]-1, num=grid_shape[0])).astype(int)

# Create a grid of coordinates using meshgrid
x, y, z = np.meshgrid(coord_range, coord_range, coord_range, indexing='ij')
x_ind, y_ind, z_ind = np.meshgrid(coord_index_range, coord_index_range, coord_index_range, indexing='ij')

del coord_index_range
gc.collect()

# Stack the coordinates to create a 2D array
coordinates_2d = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

x, y, z = x.flatten(), y.flatten(), z.flatten()
x_ind, y_ind, z_ind = x_ind.flatten(), y_ind.flatten(), z_ind.flatten()

nn = nb.NearestNeighbors(n_neighbors=1)
nn.fit(coordinates_2d)
distances, indices = nn.kneighbors(edited_coordinates)
print(f"Number of gas particles = {len(indices)}")


    # Now we make our density grid using our function of W so we can create our mask
t1 = time.time()

density_grid = np.zeros(grid_shape)

# And the indices that index into the grid_coords array
indall = np.arange(n*n*n).reshape([n,n,n])

# Now run nearest neighbours to find the index of grid_coords that is the nearest grid cell to the gas particles. 
dr = np.sqrt(3 * dx ** 3) # The length of one side in kpc is dx
density_grid = density_grid.flatten()

# We create a singular array that all the different data types will exist within
data_grid = np.zeros([10, len(density_grid)])
gas_masses = np.zeros([len(density_grid)])

#JW: sum of the weights
sum_weights = np.zeros([n,n,n]).flatten()
sum_weights2 = np.zeros([n,n,n]).flatten()

#args = [x_ind, y_ind, z_ind, indices, h_gas, dr, indall, coordinates_2d, edited_coordinates, raw_density, density_grid, raw_data, data_grid, sum_weights, sum_weights2]
#print(args)
#density_grid, data_grid, sum_weights, sum_weights2 = processing(loop_function, num_threads, args)


# for every gas particle:
for i in range(len(indices)):
    iix = x_ind[indices[i][0]]
    iiy = y_ind[indices[i][0]]
    iiz = z_ind[indices[i][0]]

    # find the number of cells that are roughly equal to h
    nn = nn_multiplier*np.ceil(h_gas[i]/dr).astype(int)    #dr is sqrt( dx*2 + dy^2 + dz^2) where the dx, dy, dz are the physical grid cell size

    # the indices of grid_coords that surround the nearest grid coordinate:
    indices_box = indall[iix-nn:iix+nn, iiy-nn:iiy+nn, iiz-nn:iiz+nn].flatten() 

    # now that you have the indices of the grid_coords, you can work out the distances of each of these to the gas particle
    # coordinates_2d and edited_coordinates are both in the same units as necessary. They are both in kpc
    box_distances = np.sqrt(np.sum((coordinates_2d[indices_box] -  edited_coordinates[i]) ** 2, axis = 1))
    ind = np.where(box_distances < 2.0*h_gas[i])[0] # JW: limiting the coordinates cuts the loop time in half
    weight = W(h_gas[i], box_distances[ind]) # We use the weight function in order to 


    density_grid[indices_box[ind]] += weight * (raw_density[i]) #* h_gas[i] ** 3 #JW: commented h_gas, changed to density
    for k in range(len(data_grid[:6])):
        data_grid[k][indices_box[ind]] += weight * (raw_data[10][i])  * raw_data[k, i] #* h_gas[i] ** 3 #JW: commented h_gas
    gas_masses[indices_box[ind]] += weight * (raw_data[10][i])  * raw_data[10, i]
    sum_weights[indices_box[ind]] += weight  #JW
    sum_weights2[indices_box[ind]] += weight * (raw_data[10][i]) #JW'''

t2 = time.time()
print('Time it took:',(t2-t1)/60., ' minutes')

#JW: remove 0's from weights
zero_mask = (sum_weights == 0)
mask = (sum_weights2 == 0) 
print(f'Lengths: sum_weights = {len(sum_weights)}, sum_weights2 = {len(sum_weights2)}, zero_mask = {len(zero_mask)}, mask = {len(mask)}')
print(np.sum(zero_mask), np.sum(mask))
if len(zero_mask != 0):
    sum_weights[zero_mask] = np.min(sum_weights[~zero_mask])/10.
if len(mask != 0):    
    sum_weights2[mask] = np.min(sum_weights2[~mask])/10.

print("Joanna Info: ")
print('Percent 0s = ',(zero_mask.sum()/(27e6))*100)
#print('Distance (kpc) to closest 0 cell = ', np.min(np.sqrt(x[mask] ** 2 + y[mask] ** 2 + z[mask] ** 2)))
#print(SubHalo['SubhaloHalfmassRadType'][4] * a / h_cosmo)

density_grid /= sum_weights #JW: divide by the weights to get units of density (10^10 Msun/kpc^3)

table_columns = []
for k in range(6): 
    data_grid[k] /= sum_weights2 #density_grid #JW: divide by sum of mass-weights to get the right units

print('Mass\nMaximums:')
print('Edited: ',np.max(np.log10(density_grid*1e10)))
print('Raw: ',np.max(np.log10(raw_data[10]*1e10)))

print('Means:')
print('Edited: ',np.mean(np.log10(density_grid[~zero_mask]*1e10)))
print('Raw: ',np.mean(np.log10(raw_data[10]*1e10)))
# This will convert everything from masses into proper densities
# First divide by the volume of each cell
# Each cell has a volume of 1 pixel cubed, and the conversion from kpc to pixels is gas_coordinates_scaling pixels / kpc
#density_grid /= (dx)**3  #JW: commented because my changes made it already a density
# Convert from 1e10 Solar Masses / kpc^3 to atoms/cm^3
density_grid *= 1e10 / 1e9 # Solar Masses / pc^3
number_density_grid = density_grid.copy() / 0.03363
density_grid /= 1.273e-17 # This factor is used by Nir to convert from Solar Masses/pc^3 to atom/cm^3

raw_density *= 1e10 #/ 1e9
raw_density /= 0.03363 

print('Density\nMaximums:')
print('Edited: ',np.max(np.log10(density_grid)))
print('Raw: ',np.max(np.log10(raw_density)))

print('Means:')
print('Edited: ',np.mean(np.log10(density_grid[~zero_mask])))
print('Raw: ',np.mean(np.log10(raw_density)))

print('Temperatures:')
print('Mean w/ 0s: ',np.mean(data_grid[0]))
print('Mean w/o 0s: ', np.mean(data_grid[0][~zero_mask]))
print('Max: ', np.max(data_grid[0]))


data_grid[6] = x #(x - (Origin[0] - shift_factor_x))
data_grid[7] = y #(y - (Origin[1] - shift_factor_y))
data_grid[8] = z #(z - (Origin[2] - shift_factor_z))
data_grid[9] += dx * 1000

#######################################################################################################################

density_grid = density_grid.reshape((n,n,n))
#position = np.meshgrid(coord_range, coord_range, coord_range, indexing='ij').reshape((n,n,n))
temperature = np.array(data_grid[0]).reshape((n,n,n))
gas_masses = gas_masses.reshape((n,n,n)) * 1e10

cool_mask = temperature <= 5e4
cool_gas = density_grid.copy()
cool_gas[~cool_mask] = 0

# Key for data_grid: Temperature, x velocity, y velocity, z velocity, SNI, SNII, x coords, y coords, z coords, cell length
good_data = {}
#gas_data['cell_size_gas'] = data_grid[9].astype('float32')
#gas_data['pos_gas'] = position
good_data['gas', 'density'] = (density_grid.astype('float32'), "g/cm**3")
good_data['gas', 'temperature'] = (temperature.astype('float32'), "K")
good_data['gas', 'mass'] = (gas_masses.astype('float32'), "Msun")
good_data['gas', 'cool_density'] = (cool_gas.astype('float64'), "g/cm**3")
baryonic_density = density_grid.copy()
#gas_data['SNII_gas'] = data_grid[5].astype('float32')
#gas_data['SNI_gas'] = data_grid[4].astype('float32')





#######################################################################################################################
gas_masses = gas_masses.reshape((n,n,n))
good_data['gas', 'mass'] = gas_masses.astype('float32')

del density_grid, temperature, gas_masses, edited_coordinates, coord_range, x, y, z, x_ind, y_ind, z_ind, coordinates_2d, v_x, v_y, v_z


#######################################################################################################################
# Do the roar (stars)
#######################################################################################################################

SubHaloStars['Coordinates'] = SubHaloStars['Coordinates'] * a / h_cosmo - Origin

mask_goodstars = ((np.abs(SubHaloStars['Coordinates'][:,0]) < HalfMassRadius3) & (np.abs(SubHaloStars['Coordinates'][:,1]) < HalfMassRadius3) &
                (np.abs(SubHaloStars['Coordinates'][:,2]) < HalfMassRadius3)) #np.std(star_coords) # We need to limit the stars radius like the gas

for key in SubHaloStars.keys():
    if key != 'count':
        if key not in ['Velocities', 'Coordinates']: SubHaloStars[key] = SubHaloStars[key][mask_goodstars]
        else: SubHaloStars[key] = SubHaloStars[key][mask_goodstars, :]

SubHaloStars['Masses'] = SubHaloStars['Masses'] / h_cosmo

#######################################################################################################################
# Vector time (again)
#######################################################################################################################
# Before we make the 3D cubic grid, we need to rotate all of the coordinates so that we look at the galaxy face-on
# Stack position and velocity components into (N, 3) arrays

edited_coordinates_stars = SubHaloStars['Coordinates'] @ R.T

#######################################################################################################################

print("Calculating Stellar Ages")
# Create a mask as any values of stellar formation time that are <=0 are wind particles, not star particles
#stellar_ages = processing(calc_stellar_ages, num_threads, SubHaloStars['GFM_StellarFormationTime'])
mask = (SubHaloStars['GFM_StellarFormationTime']>0) 
stellar_redshifts = 1/SubHaloStars['GFM_StellarFormationTime'][mask] - 1
stellar_ages =  0.95/((1+z_redshift)/7.0)**1.5 - 0.95/((1+stellar_redshifts)/7.0)**1.5
edited_coordinates_stars = edited_coordinates_stars[mask]
SubHaloStars['Masses'] = SubHaloStars['Masses'][mask] # Make sure to remove the stars with StellarFormationTime == 0
SubHaloStars['Velocities'] = SubHaloStars['Velocities'][mask, :]

#stellar_ages = (cosmo.age(z_redshift) - cosmo.age(stellar_redshifts)).value
print("Finished Stellar Ages")

# Time to remove the old stars

young_mask = stellar_ages <= 0.03
stellar_ages = stellar_ages[young_mask]
edited_coordinates_stars = edited_coordinates_stars[young_mask, :]
SubHaloStars['Masses'] = SubHaloStars['Masses'][young_mask] # Make sure to remove the stars older than 0.03Gyr
SubHaloStars['Velocities'] = SubHaloStars['Velocities'][young_mask, :]

print("Young star extent:")
print(np.min(edited_coordinates_stars, axis=0))
print(np.max(edited_coordinates_stars, axis=0))


#######################################################################################################################


# We need to use a cloud-in-cell method of smoothing the stars, so we use the nearest neighbour 8 cells to the stellar particle
# It needs to be a 2x2x2 cube around the particle, a floor-ceiling x floor ceiling x floor ceiling approach
# Mass is proportional by: (i+1)x * (j+1)y * (k+1)z for coordinates (i,j,k) where i = x.floor()


# And the indices that index into the grid_coords array
indall = np.arange(n*n*n).reshape([n,n,n])

# Now run nearest neighbours to find the index of grid_coords that is the nearest grid cell to the stellar particles.
coords = edited_coordinates_stars.copy()

coords[:,0] += HalfMassRadius3
coords[:,1] += HalfMassRadius3
coords[:,2] += HalfMassRadius3




mass_grid = np.zeros(grid_shape)
age_grid = np.zeros(grid_shape)

mass_grid = mass_grid.flatten()
age_grid = age_grid.flatten()




# grid spacing in physical coordinates
dx_cell = maxdelta / (n - 1)

# convert particle positions → cell indices (CIC step)
ix = np.floor(coords[:, 0] / dx_cell).astype(int)
iy = np.floor(coords[:, 1] / dx_cell).astype(int)
iz = np.floor(coords[:, 2] / dx_cell).astype(int)

ix = np.clip(ix, 0, n - 2)
iy = np.clip(iy, 0, n - 2)
iz = np.clip(iz, 0, n - 2)


x0 = ix * dx_cell
y0 = iy * dx_cell
z0 = iz * dx_cell

tx = (coords[:, 0] - x0) / dx_cell
ty = (coords[:, 1] - y0) / dx_cell
tz = (coords[:, 2] - z0) / dx_cell


w000 = (1 - tx) * (1 - ty) * (1 - tz)
w001 = (1 - tx) * (1 - ty) * tz
w010 = (1 - tx) * ty * (1 - tz)
w011 = (1 - tx) * ty * tz
w100 = tx * (1 - ty) * (1 - tz)
w101 = tx * (1 - ty) * tz
w110 = tx * ty * (1 - tz)
w111 = tx * ty * tz

weights = np.stack([w000, w001, w010, w011,
                    w100, w101, w110, w111], axis=1)


i000 = indall[ix,     iy,     iz]
i001 = indall[ix,     iy,     iz+1]
i010 = indall[ix,     iy+1,   iz]
i011 = indall[ix,     iy+1,   iz+1]
i100 = indall[ix+1,   iy,     iz]
i101 = indall[ix+1,   iy,     iz+1]
i110 = indall[ix+1,   iy+1,   iz]
i111 = indall[ix+1,   iy+1,   iz+1]

indices = np.stack([i000, i001, i010, i011,
                    i100, i101, i110, i111], axis=1)


flat_indices = indices.ravel()
flat_weights = weights.ravel()
flat_mass = np.repeat(SubHaloStars['Masses'], 8)
flat_age_mass = np.repeat(stellar_ages, 8) * flat_mass 
    
np.add.at(mass_grid, flat_indices, flat_mass * flat_weights)
stellar_density_grid = mass_grid / (dx_cell ** 3)

np.add.at(age_grid, flat_indices, flat_age_mass * flat_weights)


#######################################################################################################################


print(age_grid.shape, mass_grid.shape)
zero_mask = (mass_grid == 0)
age_grid[~zero_mask] = age_grid[~zero_mask]/mass_grid[~zero_mask]

print("Joanna Info: ")
print('Percent 0s = ',(zero_mask.sum()/(27e6))*100)
#print('Distance (kpc) to closest 0 cell = ', np.min(np.sqrt(x[mask] ** 2 + y[mask] ** 2 + z[mask] ** 2)))
#print(SubHalo['SubhaloHalfmassRadType'][4] * a / h_cosmo)


stellar_density_grid *= 1e10 / 1e9 # Solar Masses / pc^3
stellar_number_density_grid = stellar_density_grid.copy() / 0.03363 # This factor is used by Nir to convert from Solar Masses/pc^3 to atom/cm^3
stellar_density_grid *=  1.273e-17 # This factor converts to g/cm^3
stellar_masses = mass_grid * 1e10 # Solar Masses

#######################################################################################################################


stellar_density_grid = stellar_density_grid.reshape((n,n,n))
baryonic_density += stellar_density_grid
stellar_number_density_grid = stellar_number_density_grid.reshape((n,n,n))
age_grid = age_grid.reshape((n,n,n))
stellar_masses = stellar_masses.reshape((n,n,n))


good_data[("gas", "stellar_density")] = (stellar_density_grid.astype("float64"), "g/cm**3")
good_data[("gas", "stellar_num_density")] = (stellar_number_density_grid.astype("float32"), "1/cm**3")
good_data[("gas", "stellar_age")] = (age_grid.astype("float32"), "Gyr")
good_data[("gas", "stellar_mass")] = (stellar_masses.astype("float32"), 'Msun')
good_data[("gas", "baryonic_density")] = (baryonic_density.astype("float32"), "g/cm**3")

print("Young star extent:")
print(np.min(edited_coordinates_stars, axis=0))
print(np.max(edited_coordinates_stars, axis=0))

#######################################################################################################################

fields = ['SubhaloMassInRadType','SubhaloSFRinRad', 'SubhaloMassType', 'SubhaloHalfmassRadType', 'SubhaloGrNr']

subhalos = il.groupcat.loadSubhalos(basePath, FileNumber, fields=fields) # For some extra information

stellar_mass = np.log10(subhalos['SubhaloMassInRadType'][ID, 4]*1e10)
half_mass_r = subhalos['SubhaloHalfmassRadType'][ID, 4]


#######################################################################################################################
# Do the same for the dark matter now
#######################################################################################################################

propertiesDM = ['Velocities', 'Coordinates', 'SubfindHsml', 'SubfindDMDensity', 'ParticleIDs']
if do_dm == True:
    SubHaloDM = il.snapshot.loadSubhalo(basePath, FileNumber, id=ID, partType = particleType[2], fields=propertiesDM)
               #il.snapshot.loadSubhalo(basePath, FileNumber, id=ID, partType = particleType[1], fields=propertiesStars)
    
            
    SubHaloDM['Coordinates'] = SubHaloDM['Coordinates'] * a / h_cosmo - Origin
    SubHaloDM['SubfindDMDensity'] = SubHaloDM['SubfindDMDensity'] * h_cosmo**2 / a ** 3
    
    
    #######################################################################################################################
    # Vector time (again)
    #######################################################################################################################
    # Before we make the 3D cubic grid, we need to rotate all of the coordinates so that we look at the galaxy face-on
    # Stack position and velocity components into (N, 3) arrays
    
    # The cheeky rotation time
    edited_coordinates_dm = SubHaloDM['Coordinates'] @ R.T#_dm.T
    
    
    
    #######################################################################################################################
    # One final mask to only have stars in the desired extent
    
    # Mask the Stellar data based on position
    # We do this here a first time in order to cut out unwanted data; edited_coordinates
    # JW: some things to determine the best grid size
    # Use gas half-mass radius instead or use virial radius (maybe)
    mask_gooddm = ((np.abs(edited_coordinates_dm[:,0]) < HalfMassRadius3) & (np.abs(edited_coordinates_dm[:,1]) < HalfMassRadius3) &
                    (np.abs(edited_coordinates_dm[:,2]) < HalfMassRadius3)) #np.std(star_coords) 
    
    edited_coordinates_dm = edited_coordinates_dm[mask_gooddm]
    # ['mass', 'vel', 'pos', 'age', 'id', 'gima', 'gmmz', 'shsm']
    for key in SubHaloDM.keys():
        if key != 'count':
            if key not in ['Velocities', 'Coordinates']: SubHaloDM[key] = SubHaloDM[key][mask_gooddm]
            else: SubHaloDM[key] = SubHaloDM[key][mask_gooddm, :]
        
    print("DM extent:")
    print(np.min(edited_coordinates_dm, axis=0))
    print(np.max(edited_coordinates_dm, axis=0))
    
    # SHSML in IPM doesn't seem to align correctly to the definition expected from TNG
    
    #######################################################################################################################
    

    
    # We need to use a cloud-in-cell method of smoothing the stars, so we use the nearest neighbour 8 cells to the stellar particle
    # It needs to be a 2x2x2 cube around the particle, a floor-ceiling x floor ceiling x floor ceiling approach
    # Mass is proportional by: (i+1)x * (j+1)y * (k+1)z for coordinates (i,j,k) where i = x.floor()
    
    
    # And the indices that index into the grid_coords array
    indall = np.arange(n*n*n).reshape([n,n,n])
    
    # Now run nearest neighbours to find the index of grid_coords that is the nearest grid cell to the stellar particles.
    coords = edited_coordinates_dm.copy()
    
    coords[:,0] += HalfMassRadius3
    coords[:,1] += HalfMassRadius3
    coords[:,2] += HalfMassRadius3
    
    
    
    
    dens_grid = np.zeros(grid_shape)    
    dens_grid = dens_grid.flatten()
    
    
    
    
    # grid spacing in physical coordinates
    dx_cell = maxdelta / (n - 1)
    
    # convert particle positions → cell indices (CIC step)
    ix = np.floor(coords[:, 0] / dx_cell).astype(int)
    iy = np.floor(coords[:, 1] / dx_cell).astype(int)
    iz = np.floor(coords[:, 2] / dx_cell).astype(int)
    
    ix = np.clip(ix, 0, n - 2)
    iy = np.clip(iy, 0, n - 2)
    iz = np.clip(iz, 0, n - 2)
    
    
    x0 = ix * dx_cell
    y0 = iy * dx_cell
    z0 = iz * dx_cell
    
    tx = (coords[:, 0] - x0) / dx_cell
    ty = (coords[:, 1] - y0) / dx_cell
    tz = (coords[:, 2] - z0) / dx_cell
    
    
    w000 = (1 - tx) * (1 - ty) * (1 - tz)
    w001 = (1 - tx) * (1 - ty) * tz
    w010 = (1 - tx) * ty * (1 - tz)
    w011 = (1 - tx) * ty * tz
    w100 = tx * (1 - ty) * (1 - tz)
    w101 = tx * (1 - ty) * tz
    w110 = tx * ty * (1 - tz)
    w111 = tx * ty * tz
    
    weights = np.stack([w000, w001, w010, w011,
                        w100, w101, w110, w111], axis=1)
    
    
    i000 = indall[ix,     iy,     iz]
    i001 = indall[ix,     iy,     iz+1]
    i010 = indall[ix,     iy+1,   iz]
    i011 = indall[ix,     iy+1,   iz+1]
    i100 = indall[ix+1,   iy,     iz]
    i101 = indall[ix+1,   iy,     iz+1]
    i110 = indall[ix+1,   iy+1,   iz]
    i111 = indall[ix+1,   iy+1,   iz+1]
    
    indices = np.stack([i000, i001, i010, i011,
                        i100, i101, i110, i111], axis=1)
    
    
    flat_indices = indices.ravel()
    flat_weights = weights.ravel()
    flat_dens = np.repeat(SubHaloDM['SubfindDMDensity'], 8)
        
    np.add.at(dens_grid, flat_indices, flat_dens * flat_weights)
    
    
    #######################################################################################################################
    
    dm_density_grid = dens_grid # 10^10 Solar masses / kpc^3
    dm_mass_grid = dm_density_grid * (dx_cell ** 3)
    dm_density_grid *= 1e10 / 1e9 # Solar Masses / pc^3
    dm_density_grid /= 0.03363 # This factor is used by Nir to convert from Solar Masses/pc^3 to atom/cm^3
    dm_mass_grid *= 1e10 # Solar Masses
    
    #######################################################################################################################
    
    
    dm_density_grid = dm_density_grid.reshape((n,n,n))
    dm_mass_grid = dm_mass_grid.reshape((n,n,n))
    
    
    good_data[("gas", "dm_density")] = (dm_density_grid.astype("float32"), "1/cm**3")
    good_data[("gas", "dm_mass")] = (dm_mass_grid.astype("float32"), "Msun")
    
    

#######################################################################################################################
# Now let's start clump finding
#######################################################################################################################

# Create the ranges of coordinates and generate the yt data structure

bbox = np.array([[min_all, max_all],
                 [min_all, max_all],
                 [min_all, max_all]])

ds = yt.load_uniform_grid(good_data, stellar_density_grid.shape, bbox=bbox, length_unit="kpc")
print(f'Field List: {ds.field_list}')

del age_grid, good_data, stellar_density_grid, stellar_number_density_grid

#######################################################################################################################

print(ds)
print(ds.domain_left_edge, ds.domain_right_edge)
print(ds.domain_dimensions)

#sp = ds.all_data()
left_edge = [min_all, min_all, -4]
right_edge = [max_all, max_all, 4]
sp = ds.box(left_edge, right_edge)

print(sp[("gas", "stellar_density")].units)


'''
master_clump = Clump(sp, ("gas", "both"))
master_clump.add_validator("min_cells", 10)

c_min = 1e1
c_max = sp["gas","both"].max()
step = 2.0

print(sp["gas","both"].max())

find_clumps(master_clump, c_min, c_max, step)


#print("Found", len(leaf_clumps), "clumps")

leaf_clumps = get_leaf_clumps(master_clump)
print("Number of clumps:", len(leaf_clumps))

all_clumps = flatten_clumps(master_clump)
print("Total clumps in hierarchy:", len(all_clumps))

prj = yt.ProjectionPlot(ds, "z", ("gas", "both"), center="c", width=(20, "kpc"), data_source=sp)
prj.annotate_clumps(leaf_clumps)
prj.annotate_title(f'ID={SH_ID}; z = {z_redshift:.2f};\n'fr'$\log(M_*/M_\odot) = {stellar_mass:.2f}; R_{{*, e}} = {half_mass_r:.2f}$ kpc;')
prj.save(f"{plotsdir}IPM_{SH_ID}_baryonic_clumps")'''

prj = yt.ProjectionPlot(ds, "z", ("gas", "stellar_num_density"), center="c", width=(20, "kpc"), data_source=sp)
prj.set_cmap(field=("gas", "stellar_num_density"), cmap=cmap)
prj.annotate_title(f'ID={ID}; z = {z_redshift:.2f};'fr'$\log(M_*/M_\odot) = {stellar_mass:.2f}; R_{{*, e}} = {half_mass_r:.2f}$ kpc;')
prj.save(f"{plotsdir}TNG_{ID}")

prj = yt.ProjectionPlot(ds, "z", ("gas", "density"), center="c", width=(20, "kpc"), data_source=sp)
prj.set_cmap(field=("gas", "density"), cmap=cmap)
prj.annotate_title(f'ID={ID}; z = {z_redshift:.2f};'fr'$\log(M_*/M_\odot) = {stellar_mass:.2f}; R_{{*, e}} = {half_mass_r:.2f}$ kpc;')
prj.save(f"{plotsdir}TNG_{ID}")

prj = yt.ProjectionPlot(ds, "z", ("gas", "baryonic_density"), center="c", width=(20, "kpc"), data_source=sp)
prj.set_cmap(field=("gas", "baryonic_density"), cmap=cmap)
prj.annotate_title(f'ID={ID}; z = {z_redshift:.2f};'fr'$\log(M_*/M_\odot) = {stellar_mass:.2f}; R_{{*, e}} = {half_mass_r:.2f}$ kpc;')
prj.save(f"{plotsdir}TNG_{ID}")

master_clump = Clump(sp, ("gas", "stellar_num_density"))
#master_clump.add_validator("max_cells", 100)
master_clump.add_validator("min_cells", 10)

c_min = 10 ** np.floor(np.log10(sp["gas","stellar_num_density"].max()) - 3)
c_max = 10 ** np.floor(np.log10(sp["gas","stellar_num_density"].max()) + 1)
step = 2.0

print(sp["gas","stellar_density"].max())


find_clumps(master_clump, c_min, c_max, step)
fn = master_clump.save_as_dataset(f"{clumpsdir}TNG_{ID}_clumps", fields=[("gas", "stellar_num_density")])

#print("Found", len(leaf_clumps), "clumps")

leaf_clumps = get_leaf_clumps(master_clump)
print("Number of clumps:", len(leaf_clumps))

clumps_mask = []

for clump in leaf_clumps:
    clumps_mask.append(clump['gas','dm_density'].mean() < clump_dm_limit and 
                        clump['gas', "stellar_num_density"].max() > clump_dens_min)

#sourc
prj = yt.ProjectionPlot(ds, "z", ("gas", "stellar_num_density"), center="c", width=(20, "kpc"), data_source=sp)
prj.annotate_clumps(np.array(leaf_clumps)[clumps_mask])
prj.annotate_title(f'ID={ID}; z = {z_redshift:.2f};'fr'$\log(M_*/M_\odot) = {stellar_mass:.2f}; R_{{*, e}} = {half_mass_r:.2f}$ kpc;')
prj.save(f"{plotsdir}TNG_{ID}_Clumps")


prj = yt.ProjectionPlot(ds, "z", ("gas", "density"), center="c", width=(20, "kpc"), data_source=sp)
prj.annotate_clumps(np.array(leaf_clumps)[clumps_mask])
prj.annotate_title(f'ID={ID}; z = {z_redshift:.2f};'fr'$\log(M_*/M_\odot) = {stellar_mass:.2f}; R_{{*, e}} = {half_mass_r:.2f}$ kpc;')
prj.save(f"{plotsdir}TNG_{ID}_Clumps")




with open(f"{plotsdir}TNG_{ID}_clump_properties.txt", "w") as f:
    f.write(f"{'potential_clump'}, {'clump_type'}, {'clump_id':8}, {'n_cells':7}, "
            f" log10 Mass, x (kpc), y (kpc), z (kpc), "
            f"{'max_density':>19}, {'mean_density':>19}, {'mean_dm_density':>19}, "
            f"mean_stellar_age\n")

    for i, clump in enumerate(leaf_clumps):
        
        potential_clump = str(clumps_mask[i])

        n_cells = clump["gas", "stellar_density"].size

        mass = np.log10(clump.quantities.total_quantity(("gas", "stellar_mass")).to_value("Msun"))

        center = clump.quantities.center_of_mass().to("kpc").value

        max_density = clump["gas", "stellar_num_density"].max()
        
        mean_density = clump['gas', "stellar_num_density"].mean()
        
        dm_density = clump['gas', 'dm_density'].mean()
        
        age = clump['gas', 'stellar_age'].mean()
        
        if np.sqrt(center[0] ** 2 + center[1] ** 2 + center[2] ** 2) <= 0.1:
            clump_type = 'bulge'
        elif (clump['gas','dm_density'].mean() < clump_dm_limit):
            clump_type = 'in_situ'
        else:
            clump_type = 'ex_situ'

        f.write(
            f"{potential_clump:>15}, {clump_type:>10}, {i:8}, {n_cells:7}, {mass:.8f}, "
            f"{center[0]:7.4f}, {center[1]:7.4f}, {center[2]:7.4f}, "
            f"{max_density:10.3f}, {mean_density:10.3f}, {dm_density:10.3f}, "
            f"{age:12.3f}\n"
            )
        
print("Finished")

