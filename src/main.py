# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:40:59 2026

@author: MalcolmP
"""

import data as dt
from plotting import plot_gas, plot_stars, plot_dm
import tomllib


def run_tng50_visualizer(config_file):
    
    with open(config_file, "rb") as f:
        config = tomllib.load(f)
        
        snapshot        = config['snapshot']
        subhalo_ID      = config['subhalo_ID']
        
        data_path       = config['data_path']
        plot_path       = config['plot_path']
        
        radius          = config['radius']
        resolution      = config['resolution']
        nn_multiplier   = config['nn_multiplier']
        
        plotting_axis   = config['plotting_axis']
        cmap            = config['cmap']
        
        
        
        ################################################################################################################        
        
        
        origin, z_redshift, scale_factor, h_cosmo = dt.load_subhalo_info(data_path, subhalo_ID, snapshot)
        
        
        subhalo_gas_physical    = dt.load_subhalo_gas(
            data_path, subhalo_ID, snapshot, origin, scale_factor, h_cosmo)
        
        subhalo_stars_physical  = dt.load_subhalo_stars(
            data_path, subhalo_ID, snapshot, origin, z_redshift, scale_factor, h_cosmo)
        
        subhalo_dm_physical     = dt.load_subhalo_dm(
            data_path, subhalo_ID, snapshot, origin, scale_factor, h_cosmo)
        
        ################################################################################################################
        
        rotational_matrix = dt.rotational_matrix(
            subhalo_gas_physical['Coordinates'], subhalo_gas_physical['Masses'], subhalo_gas_physical['Velocities'])
       
        
        subhalo_gas_physical['Coordinates']     = subhalo_gas_physical['Coordinates'] @ rotational_matrix.T
        subhalo_gas_physical['Velocities']      = subhalo_gas_physical['Velocities'] @ rotational_matrix.T
        
        subhalo_stars_physical['Coordinates']   = subhalo_stars_physical['Coordinates'] @ rotational_matrix.T
        subhalo_stars_physical['Velocities']    = subhalo_stars_physical['Velocities'] @ rotational_matrix.T
        
        subhalo_dm_physical['Coordinates']      = subhalo_dm_physical['Coordinates'] @ rotational_matrix.T
        subhalo_dm_physical['Velocities']       = subhalo_dm_physical['Velocities'] @ rotational_matrix.T
        
        ################################################################################################################
        
        subhalo_gas_physical_grid, subhalo_cool_gas_physical_grid = (
            dt.gas_data_grid_conversion(subhalo_gas_physical, nn_multiplier, resolution, radius=15)
            )
        
        subhalo_stars_physical_grid   = (
            dt.star_data_grid_conversion(subhalo_stars_physical, resolution, radius)
            )
        
        subhalo_dm_physical_grid    = (
            dt.dm_data_grid_conversion(subhalo_dm_physical, resolution, radius)
            )
        
        ################################################################################################################
        
        plot_gas(plot_path, subhalo_ID, snapshot, subhalo_gas_physical_grid, radius, cmap, plotting_axis)
        
        plot_stars(plot_path, subhalo_ID, snapshot, subhalo_stars_physical_grid, radius, cmap, plotting_axis)
        
        plot_dm(plot_path, subhalo_ID, snapshot, subhalo_dm_physical_grid, radius, cmap, plotting_axis)
        
    
    return()