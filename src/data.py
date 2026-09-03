# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:41:02 2026

@author: MalcolmP
"""

import numpy as np
import scipy.constants as const
from astropy.cosmology import Planck15 as cosmo
import illustris_python as il
import sklearn.neighbors as nb



def load_subhalo_info(data_path,
                      subhalo_ID,
                      snapshot
                      ):
    
    
    subhalo = il.snapshot.loadSingle(data_path, snapshot, subhaloID=subhalo_ID)
    header = il.groupcat.loadHeader(data_path, snapshot)
    
    #z_redshift = 3 # JW: renamed z (z is used as a coordinate later)
    z_redshift = header["Redshift"]
    scale_factor = 1 / (1 + z_redshift)
    h_cosmo = cosmo.H(0).value / 100

    origin = subhalo['SubhaloPos'] * scale_factor / h_cosmo
    
    return(origin, z_redshift, scale_factor, h_cosmo)


def load_subhalo_gas(
        data_path,
        subhalo_ID,
        snapshot,
        origin,
        scale_factor,
        h_cosmo,
        radius=15                   # kpc
        ):
    
    properties_gas = [
        'Masses', 'Velocities', 'Coordinates', 'Density',
        'SubfindHsml', 'InternalEnergy', 'GFM_MetalsTagged', 'ElectronAbundance'
        ]
    
    subhalo_gas = il.snapshot.loadSubhalo(
        data_path, snapshot, id=subhalo_ID, partType='gas', fields=properties_gas
        )
    
    subhalo_gas_physical = {}
    
    
    # We now put the coordinates into units of kpc from ckpc/h
    subhalo_gas_physical['Coordinates'] = subhalo_gas['Coordinates'] * scale_factor / h_cosmo - origin                    # kpc
    particle_radii = np.linalg.norm(subhalo_gas_physical['Coordinates'], axis=1)
    # Remove any unwanted data beyond the specified radius
    mask_goodgas = (particle_radii <= radius)
    subhalo_gas_physical['Coordinates'] = subhalo_gas_physical['Coordinates'][mask_goodgas]
    
    
    # Convert remaining quantities to physical units
    subhalo_gas_physical['Masses'] = subhalo_gas['Masses'][mask_goodgas] * 1e10 / h_cosmo                               # M_sun
    
    subhalo_gas_physical['Density'] = subhalo_gas['Density'][mask_goodgas] * 1e10 * (h_cosmo ** 2 / scale_factor ** 3)  # M_sun/kpc^3
    
    subhalo_gas_physical['Velocities'] = subhalo_gas['Velocities'][mask_goodgas, :] * np.sqrt(scale_factor)             # km/s
    
    subhalo_gas_physical['SubfindHsml'] = subhalo_gas['SubfindHsml'][mask_goodgas] * scale_factor / h_cosmo             # kpc


    # Here is where we calculate our smoothing radius, h
    particle_volumes = subhalo_gas_physical['Masses'] / subhalo_gas_physical['Density']
    factor = 1.0
    h_smoothing = factor * (
        (0.75 * particle_volumes / np.pi) ** (1 / 3)
        ) #JW: factor
    subhalo_gas_physical['h_smoothing'] = h_smoothing
    
    
    # And finally we caculate the temperature
    internal_energy = subhalo_gas['InternalEnergy'] * 1e6 # m^2/s^2
    xe = subhalo_gas['ElectronAbundance']
    Xh = 0.76
    gamma = 5/3
    kb = const.k #kb in J/K
    mu = 4 * const.m_p / (1 + 3 * Xh + 4 * Xh * xe) # units of kg
    subhalo_gas_physical['Temperature'] = (gamma - 1) * (internal_energy / kb) * mu

    return(subhalo_gas_physical)



def load_subhalo_stars(
        data_path,
        subhalo_ID,
        snapshot,
        origin,
        z_redshift,
        scale_factor,
        h_cosmo,
        radius=15,                  # kpc
        exact_stellar_ages=False
        ):
    
    properties_stars = [
        'Masses', 'Velocities', 'Coordinates', 'StellarHsml',
        'GFM_StellarFormationTime', 'GFM_InitialMass',
        ]
    
    subhalo_stars = il.snapshot.loadSubhalo(
        data_path, snapshot, id=subhalo_ID, partType='stars', fields=properties_stars
        )
    
    subhalo_stars_physical = {}
    
    
    # We now put the coordinates into units of kpc from ckpc/h
    subhalo_stars_physical['Coordinates'] = subhalo_stars['Coordinates'] * scale_factor / h_cosmo - origin                # kpc
    particle_radii = np.linalg.norm(subhalo_stars_physical['Coordinates'], axis=1)
    # Remove any unwanted data beyond the specified radius and with a stellar age of 0
    mask_good_stars = (particle_radii <= radius) & (subhalo_stars['GFM_StellarFormationTime'] > 0)
    subhalo_stars_physical['Coordinates'] = subhalo_stars_physical['Coordinates'][mask_good_stars]
    
    
    # Convert remaining quantities to physical units
    subhalo_stars_physical['Masses'] = subhalo_stars['Masses'][mask_good_stars] * 1e10 / h_cosmo                        # M_sun
    
    subhalo_stars_physical['GFM_InitialMass'] = subhalo_stars['GFM_InitialMass'][mask_good_stars] * 1e10 / h_cosmo      # M_sun
    
    subhalo_stars_physical['Velocities'] = subhalo_stars['Velocities'][mask_good_stars, :] * np.sqrt(scale_factor)      # km/s
    
    subhalo_stars_physical['StellarHsml'] = subhalo_stars['StellarHsml'][mask_good_stars] * scale_factor / h_cosmo      # kpc
    
    subhalo_stars_physical['GFM_StellarFormationTime'] = subhalo_stars['GFM_StellarFormationTime'][mask_good_stars]

    
    # We now need to calculate stellar ages
    stellar_redshifts = 1 / subhalo_stars_physical['GFM_StellarFormationTime'] - 1
    if exact_stellar_ages:
        formation_times = cosmo.age(stellar_redshifts)
        current_time = cosmo.age(z_redshift)
        
        subhalo_stars_physical['stellar_ages'] = (current_time - formation_times).value                                 # Gyr
        
    else:
        subhalo_stars_physical['stellar_ages'] = (
            0.95 / ((1 + z_redshift) / 7.0) ** 1.5 - 0.95 / 
            ((1 + stellar_redshifts) / 7.0) ** 1.5)                                       # Gyr


    return(subhalo_stars_physical)


def load_subhalo_dm(
        data_path,
        subhalo_ID,
        snapshot,
        origin,
        scale_factor,
        h_cosmo,
        radius=15                   # kpc
        ):
    
    
    properties_dm = [
        'Velocities', 'Coordinates', 'SubfindDMDensity'
        ]
    
    subhalo_dm = il.snapshot.loadSubhalo(
        data_path, snapshot, id=subhalo_ID, partType = 'dm', fields=properties_dm
        )
    
    subhalo_dm_physical = {}
    
    
    # We now put the coordinates into units of kpc from ckpc/h
    subhalo_dm_physical['Coordinates'] = (
        subhalo_dm['Coordinates'] * 
        scale_factor / h_cosmo - origin)                                                  # kpc
    
    particle_radii = np.linalg.norm(
        subhalo_dm_physical['Coordinates'], axis=1)
    
    # Remove any unwanted data beyond the specified radius
    mask_good_dm = (particle_radii <= radius)
    subhalo_dm_physical['Coordinates'] = subhalo_dm_physical['Coordinates'][mask_good_dm]
    
            
    subhalo_dm_physical['SubfindDMDensity'] = (
        subhalo_dm['SubfindDMDensity'][mask_good_dm] * 
        1e10 * h_cosmo ** 2 / scale_factor ** 3)                                                                        # M_sun/kpc^3
    
    subhalo_dm_physical['Velocities'] = (
        subhalo_dm['Velocities'][mask_good_dm, :]
        * np.sqrt(scale_factor))                                                          # km/s
    
    
    return(subhalo_dm_physical)


def rotational_matrix(
        coordinates,
        masses,
        velocities
        ):
    
    
    # Densities can be used in place of masses if no mass data are present
    # Cross product for each particle
    L = np.cross(coordinates, velocities)   # shape (N, 3)
    L_total = np.sum(L * masses[:, None], axis = 0)
    L_hat = L_total / np.linalg.norm(L_total)

    z_axis = np.array([0.0, 0.0, 1.0])
    
    L_hat = L_hat / np.linalg.norm(L_hat)
    z_axis = z_axis / np.linalg.norm(z_axis)
    v = np.cross(L_hat, z_axis)
    c = np.dot(L_hat, z_axis)
    if np.allclose(c, 1.0):
        return np.eye(3)   # already aligned
    if np.allclose(c, -1.0):
        # 180-degree rotation: need an arbitrary perpendicular axis
        # choose axis orthogonal to a
        axis = np.array([1.0, 0.0, 0.0])
        if np.allclose(np.abs(L_hat), axis):
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(L_hat, axis)
        v = v / np.linalg.norm(v)
        return -np.eye(3) + 2.0 * np.outer(v, v)
    s = np.linalg.norm(v)
    kmat = np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],     0]])
    R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))
    return(R)  


def gas_data_grid_conversion(
        subhalo_gas_physical,
        nn_multiplier = 3,
        resolution=0.1,             # kpc       
        radius=15,                  # kpc
        ):
    
    
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
 
    
    # Now we need to fit the coordinates to our grid using the nearest neighbors
    n = int(2 * radius / resolution) + 1 # This is the dimension of our cubic grid
    grid_shape = (n, n, n)
    
    # We now need to initialize our cubic grids
    # Start by initiating our data grids
    subhalo_gas_physical_grid = {}
    subhalo_gas_physical_grid['gas', 'mass'] = np.zeros(n ** 3)
    subhalo_gas_physical_grid['gas', 'density'] = np.zeros(n ** 3)
    subhalo_gas_physical_grid['gas', 'temperature'] = np.zeros(n ** 3)
    subhalo_gas_physical_grid['gas', 'velocity'] = np.zeros(n ** 3)
    subhalo_gas_physical_grid['gas', 'velocity_x'] = np.zeros(n ** 3)
    subhalo_gas_physical_grid['gas', 'velocity_y'] = np.zeros(n ** 3)
    subhalo_gas_physical_grid['gas', 'velocity_z'] = np.zeros(n ** 3)

    # Coordinates
    min_all, max_all = -radius, radius
    coord_range = np.linspace(min_all,max_all,n).astype('float32') #JW: make all directions go from min_all to max_all

    # Indices of Coordinates
    coord_index_range = (np.linspace(0, grid_shape[0]-1, num=grid_shape[0])).astype(int)
    #y_index_range = (np.linspace(0, grid_shape[0]-1, num=grid_shape[0])).astype(int)
    #z_index_range = (np.linspace(0, grid_shape[0]-1, num=grid_shape[0])).astype(int)

    # Create a grid of coordinates using meshgrid
    x, y, z = np.meshgrid(coord_range, coord_range, coord_range, indexing='ij')
    x_ind, y_ind, z_ind = np.meshgrid(coord_index_range, coord_index_range, coord_index_range, indexing='ij')


    # Stack the coordinates to create a 2D array
    coordinates_2d = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    x, y, z = x.flatten(), y.flatten(), z.flatten()
    x_ind, y_ind, z_ind = x_ind.flatten(), y_ind.flatten(), z_ind.flatten()

    nn = nb.NearestNeighbors(n_neighbors=1)
    nn.fit(coordinates_2d)
    distances, indices = nn.kneighbors(subhalo_gas_physical['Coordinates'])
    print(f"Number of gas particles = {len(indices)}")


    # Create the indices that index into the grid_coords array
    indall = np.arange(n*n*n).reshape([n,n,n])

    # Now run nearest neighbours to find the index of grid_coords that is the nearest grid cell to the gas particles. 
    dr = np.sqrt(3 * resolution ** 3) # The length of one side in kpc is resolution


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
        nn = nn_multiplier * np.ceil(subhalo_gas_physical['h_smoothing'][i] / dr).astype(int)    #dr is sqrt( dx*2 + dy^2 + dz^2) where the dx, dy, dz are the physical grid cell size

        # the indices of grid_coords that surround the nearest grid coordinate:
        indices_box = indall[iix - nn:iix + nn, iiy - nn:iiy + nn, iiz - nn:iiz + nn].flatten() 

        # now that you have the indices of the grid_coords, you can work out the distances of each of these to the gas particle
        # coordinates_2d and edited_coordinates are both in the same units as necessary. They are both in kpc
        box_distances = np.sqrt(np.sum((coordinates_2d[indices_box] -
                                        subhalo_gas_physical['Coordinates'][i]) ** 2, axis = 1))
        ind = np.where(box_distances < 2.0 * subhalo_gas_physical['h_smoothing'][i])[0] # JW: limiting the coordinates cuts the loop time in half
        weight = W(subhalo_gas_physical['h_smoothing'][i], box_distances[ind]) # We use the weight function in order to 


        subhalo_gas_physical_grid['gas', 'density'][indices_box[ind]] +=(
            weight * (subhalo_gas_physical['Density'][i]))
        
        subhalo_gas_physical_grid['gas', 'mass'][indices_box[ind]] += (
            weight * subhalo_gas_physical['Masses'][i])
        
        for axis_num, axis in enumerate(['x', 'y', 'z']):        
            subhalo_gas_physical_grid['gas', f'velocity_{axis}'][indices_box[ind]] += (
                weight * (subhalo_gas_physical['Masses'][i] *
                          subhalo_gas_physical['Velocities'][i, axis_num]))        
        
        subhalo_gas_physical_grid['gas', 'temperature'][indices_box[ind]] += weight * (
            subhalo_gas_physical['Masses'][i] * subhalo_gas_physical['Temperature'][i])
        
        
        sum_weights[indices_box[ind]] += weight
        sum_weights2[indices_box[ind]] += weight * (subhalo_gas_physical['Masses'][i])



    # Remove 0's from weights
    zero_mask = (sum_weights == 0)
    mask = (sum_weights2 == 0) 
    print(f'Lengths:'
          f' sum_weights = {len(sum_weights)},'
          f' sum_weights2 = {len(sum_weights2)},'
          f' zero_mask = {len(zero_mask)},'
          f' mask = {len(mask)}')
    
    print(np.sum(zero_mask), np.sum(mask))
    if len(zero_mask != 0):
        sum_weights[zero_mask] = np.min(sum_weights[~zero_mask])/10.
    if len(mask != 0):    
        sum_weights2[mask] = np.min(sum_weights2[~mask])/10.

    print('Percent 0s = ',(zero_mask.sum()/(27e6))*100)

    subhalo_gas_physical_grid['gas', 'density'] /= sum_weights                                                          # M_sun/kpc
    
    subhalo_gas_physical_grid['gas', 'mass'] /= n ** 3                                                                  # M_sun
    
    for axis in ['x', 'y', 'z']:
        subhalo_gas_physical_grid['gas', f'velocity_{axis}'] /= sum_weights2                                                         # km/s
    
    subhalo_gas_physical_grid['gas', 'temperature'] /= sum_weights2                                                     # K

    
    subhalo_gas_physical_grid['gas', 'velocity'] = np.sqrt(
        subhalo_gas_physical_grid['gas', 'velocity_x'] ** 2 +
        subhalo_gas_physical_grid['gas', 'velocity_y'] ** 2 +
        subhalo_gas_physical_grid['gas', 'velocity_z'] ** 2)

    ####################################################################################################################

    subhalo_gas_physical_grid['gas', 'density']     = (
        subhalo_gas_physical_grid['gas', 'density'].reshape(
            (n,n,n)).astype('float32'), "Msun/kpc**3")
    subhalo_gas_physical_grid['gas', 'mass']        = (
        subhalo_gas_physical_grid['gas', 'mass'].reshape(
            (n,n,n)).astype('float32'), "Msun")
    subhalo_gas_physical_grid['gas', 'velocity']    = (
        subhalo_gas_physical_grid['gas', 'velocity'].reshape(
            (n,n,n)).astype('float32'), "km/s")
    subhalo_gas_physical_grid['gas', 'temperature'] = (
        subhalo_gas_physical_grid['gas', 'temperature'].reshape(
            (n,n,n)).astype('float32'), "K")
    
    for axis in ['x', 'y', 'z']:
        subhalo_gas_physical_grid['gas', f'velocity_{axis}']    = (
            subhalo_gas_physical_grid['gas', f'velocity_{axis}'].reshape(
                (n,n,n)).astype('float32'), "km/s")
    
    
    subhalo_cool_gas_physical_grid = subhalo_gas_physical_grid.copy()

    hot_mask = (subhalo_gas_physical_grid['gas', 'temperature'] > 5e4)
    
    subhalo_cool_gas_physical_grid['gas', 'density'][hot_mask]      = 0
    subhalo_cool_gas_physical_grid['gas', 'mass'][hot_mask]         = 0
    subhalo_cool_gas_physical_grid['gas', 'velocity'][hot_mask]     = 0
    subhalo_cool_gas_physical_grid['gas', 'temperature'][hot_mask]  = 0
    for axis in ['x', 'y', 'z']:
        subhalo_cool_gas_physical_grid['gas', f'velocity_{axis}'][hot_mask]  = 0
    
    return(subhalo_gas_physical_grid, subhalo_cool_gas_physical_grid)


def star_data_grid_conversion(
        subhalo_stars_physical,
        resolution=0.1,
        radius=15,
        ):
    
    
    def cloud_in_cell(raw_stars, grid_stars, resolution, n, radius):
        # We need to use a cloud-in-cell method of smoothing the stars, so we use the nearest neighbour 8 cells to the stellar particle
        # It needs to be a 2x2x2 cube around the particle, a floor-ceiling x floor ceiling x floor ceiling approach
        # Mass is proportional by: (i+1)x * (j+1)y * (k+1)z for coordinates (i,j,k) where i = x.floor()
        
        

        indall = np.arange(n*n*n).reshape([n,n,n])
        
        coords = raw_stars['Coordinates'].copy()
        
        coords[:,0] += radius
        coords[:,1] += radius
        coords[:,2] += radius    
        

        
        # convert particle positions → cell indices (CIC step)
        ix = np.floor(coords[:, 0] / resolution).astype(int)
        iy = np.floor(coords[:, 1] / resolution).astype(int)
        iz = np.floor(coords[:, 2] / resolution).astype(int)
        
        ix = np.clip(ix, 0, n - 2)
        iy = np.clip(iy, 0, n - 2)
        iz = np.clip(iz, 0, n - 2)
        
        
        x0 = ix * resolution
        y0 = iy * resolution
        z0 = iz * resolution
        
        tx = (coords[:, 0] - x0) / resolution
        ty = (coords[:, 1] - y0) / resolution
        tz = (coords[:, 2] - z0) / resolution
        
        
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
        flat_mass = np.repeat(raw_stars['Masses'], 8)
        flat_age_mass = np.repeat(raw_stars['stellar_ages'], 8) * flat_mass
        flat_velocity_mass = np.repeat(raw_stars['Velocities'], 8) * flat_mass
            
        np.add.at(grid_stars['stars', 'mass'], flat_indices, flat_mass * flat_weights)
        grid_stars['stars', 'density'] = grid_stars['stars', 'mass'] / (resolution ** 3)
        
        np.add.at(grid_stars['stars', 'age'], flat_indices, flat_age_mass * flat_weights)
        for axis_num, axis in enumerate(['x', 'y', 'z']):
            np.add.at(grid_stars['stars', f'velocity_{axis}'],
                      flat_indices, 
                      flat_velocity_mass[axis_num] * flat_weights)
        
        # Divide by weights
        zero_mask = (grid_stars['stars', 'mass'] == 0)
        grid_stars['stars', 'age'][~zero_mask] /= grid_stars['stars', 'mass'][~zero_mask]
        for axis in ['x', 'y', 'z']:
            grid_stars['stars', f'velocity_{axis}'][~zero_mask] /= (
                grid_stars['stars', 'mass'][~zero_mask])
        
        return(grid_stars)
        
    ####################################################################################################################  
        
    
    n = int(2 * radius / resolution) + 1 # This is the dimension of our cubic grid
    
    # First we need to initialize our Stellar Grid for all stars and young stars
    # And make a dictionary of the raw data for young stars
    
    subhalo_stars_physical_grid = {}
    subhalo_stars_physical_grid['stars', 'mass']        = np.zeros(n ** 3)
    subhalo_stars_physical_grid['stars', 'velocity']    = np.zeros(n ** 3)
    subhalo_stars_physical_grid['stars', 'velocity_x']  = np.zeros(n ** 3)
    subhalo_stars_physical_grid['stars', 'velocity_y']  = np.zeros(n ** 3)
    subhalo_stars_physical_grid['stars', 'velocity_z']  = np.zeros(n ** 3)
    subhalo_stars_physical_grid['stars', 'density']     = np.zeros(n ** 3)
    subhalo_stars_physical_grid['stars', 'age']         = np.zeros(n ** 3)
    
   
    ####################################################################################################################
    
    subhalo_stars_physical_grid = cloud_in_cell(subhalo_stars_physical,
                                                subhalo_stars_physical_grid,
                                                resolution,
                                                n, radius)
    
    subhalo_stars_physical_grid['stars', 'velocity'] = np.sqrt(
        subhalo_stars_physical_grid['stars', 'velocity_x'] ** 2 +
        subhalo_stars_physical_grid['stars', 'velocity_y'] ** 2 +
        subhalo_stars_physical_grid['stars', 'velocity_z'] ** 2)
    
    ####################################################################################################################
    
      
    subhalo_stars_physical_grid['stars', 'density']  = (
        subhalo_stars_physical_grid['stars', 'density'].reshape(
            (n,n,n)).astype('float32'), "Msun/kpc**3")
    subhalo_stars_physical_grid['stars', 'age']      = (
        subhalo_stars_physical_grid['stars', 'age'].reshape(
            (n,n,n)).astype('float32'), "Gyr")
    subhalo_stars_physical_grid['stars', 'mass']     = (
        subhalo_stars_physical_grid['stars', 'mass'].reshape(
            (n,n,n)).astype('float32'), "Msun")
    subhalo_stars_physical_grid['stars', 'velocity'] = (
        subhalo_stars_physical_grid['stars', 'velocity'].reshape(
            (n,n,n)).astype('float32'), "km/s")
    
    for axis in ['x', 'y', 'z']:
        subhalo_stars_physical_grid['stars', f'velocity_{axis}'] = (
            subhalo_stars_physical_grid['stars', f'velocity_{axis}'].reshape(
                (n,n,n)).astype('float32'), "km/s")
        
    
        
    ####################################################################################################################
    
    return(subhalo_stars_physical_grid)  



def dm_data_grid_conversion(
        subhalo_dm_physical,
        resolution=0.1,
        radius=15,
        ):
    
    
    def cloud_in_cell(raw_dm, grid_dm, resolution, n, radius):
        # We need to use a cloud-in-cell method of smoothing the stars, so we use the nearest neighbour 8 cells to the stellar particle
        # It needs to be a 2x2x2 cube around the particle, a floor-ceiling x floor ceiling x floor ceiling approach
        # Mass is proportional by: (i+1)x * (j+1)y * (k+1)z for coordinates (i,j,k) where i = x.floor()
        
        

        indall = np.arange(n*n*n).reshape([n,n,n])
        
        coords = raw_dm['Coordinates'].copy()
        
        coords[:,0] += radius
        coords[:,1] += radius
        coords[:,2] += radius    
        

        
        # convert particle positions → cell indices (CIC step)
        ix = np.floor(coords[:, 0] / resolution).astype(int)
        iy = np.floor(coords[:, 1] / resolution).astype(int)
        iz = np.floor(coords[:, 2] / resolution).astype(int)
        
        ix = np.clip(ix, 0, n - 2)
        iy = np.clip(iy, 0, n - 2)
        iz = np.clip(iz, 0, n - 2)
        
        
        x0 = ix * resolution
        y0 = iy * resolution
        z0 = iz * resolution
        
        tx = (coords[:, 0] - x0) / resolution
        ty = (coords[:, 1] - y0) / resolution
        tz = (coords[:, 2] - z0) / resolution
        
        
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
        flat_dens = np.repeat(raw_dm['SubfindDMDensity'], 8)
        flat_velocity_density = np.repeat(raw_dm['Velocities'], 8) * flat_dens
        
        np.add.at(grid_dm['dm', 'density'], flat_indices, flat_dens * flat_weights)
        grid_dm['dm', 'mass'] = grid_dm['dm', 'density'] * (resolution ** 3)
        for axis_num, axis in enumerate(['x', 'y', 'z']):
            np.add.at(grid_dm['dm', f'velocity{axis}'],
                      flat_indices,
                      flat_velocity_density[axis_num] * flat_weights)
        
        zero_mask = grid_dm['dm', 'density'] == 0
        for axis in ['x', 'y', 'z']:
            grid_dm['dm', f'velocity_{axis}'][~zero_mask] /= (
                grid_dm['dm', 'density'][~zero_mask])
         
        return(grid_dm)
        
    ####################################################################################################################  
        
    
    n = int(2 * radius / resolution) + 1 # This is the dimension of our cubic grid
    
    # First we need to initialize our Stellar Grid for all stars and young stars
    # And make a dictionary of the raw data for young stars
    
    subhalo_dm_physical_grid = {}
    subhalo_dm_physical_grid['dm', 'density']    = np.zeros(n ** 3)
    subhalo_dm_physical_grid['dm', 'velocity']   = np.zeros(n ** 3)
    subhalo_dm_physical_grid['dm', 'velocity_x'] = np.zeros(n ** 3)
    subhalo_dm_physical_grid['dm', 'velocity_y'] = np.zeros(n ** 3)
    subhalo_dm_physical_grid['dm', 'velocity_z'] = np.zeros(n ** 3)
    subhalo_dm_physical_grid['dm', 'mass']       = np.zeros(n ** 3)
    
   
    ####################################################################################################################
    
    subhalo_dm_physical_grid = cloud_in_cell(subhalo_dm_physical,
                                                subhalo_dm_physical_grid,
                                                resolution,
                                                n, radius)
    
    subhalo_dm_physical_grid['dm', 'velocity'] = np.sqrt(
        subhalo_dm_physical_grid['dm', 'velocity_x'] ** 2 +
        subhalo_dm_physical_grid['dm', 'velocity_y'] ** 2 +
        subhalo_dm_physical_grid['dm', 'velocity_z'] ** 2)
    
    ####################################################################################################################
    
    subhalo_dm_physical_grid['dm', 'mass']     = (
        subhalo_dm_physical_grid['dm', 'mass'].reshape(
            (n,n,n)).astype('float32'), "Msun/kpc**3")
    subhalo_dm_physical_grid['dm', 'density']  = (
        subhalo_dm_physical_grid['dm', 'density'].reshape(
            (n,n,n)).astype('float32'), "Msun")
    subhalo_dm_physical_grid['dm', 'velocity'] = (
        subhalo_dm_physical_grid['dm', 'velocity'].reshape(
            (n,n,n)).astype('float32'), "km/s")
    
    for axis in ['x', 'y', 'z']:
        subhalo_dm_physical_grid['dm', f'velocity_{axis}']       = (
            subhalo_dm_physical_grid['dm', f'velocity_{axis}'].reshape(
                (n,n,n)).astype('float32'), "km/s")
        
    ####################################################################################################################
    
    return(subhalo_dm_physical_grid)  