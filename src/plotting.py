# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:41:30 2026

@author: MalcolmP
"""


import numpy as np
import yt


def generate_plots(
        plot_path,
        data_source,
        sp,
        particle_type,
        field,
        subhalo_ID,
        radius,
        axis,
        cmap):
    
    prj = yt.ProjectionPlot(data_source, "z", field, center="c", width=(20, "kpc"), data_source=sp)
    prj.set_cmap(field, cmap=cmap)
    prj.annotate_title(f'Subhalo ID: {subhalo_ID}; {particle_type}_{field[-1]}')
    prj.save(f"{plot_path}/TNG-{subhalo_ID}_{particle_type}_{field[-1]}")
    print(data_source.length_unit)
    print(data_source.unit_system)
    
    return()


def plot_gas(plot_path,
             subhalo_ID,
             snapshot,
             subhalo_gas_physical_grid,
             radius=15,
             cmap='arbre',
             axis='z'):
    
    bbox = np.array([[-radius, radius],
                     [-radius, radius],
                     [-radius, radius]])

    data_source = yt.load_uniform_grid(
        subhalo_gas_physical_grid,
        subhalo_gas_physical_grid[('gas', 'density')][0].shape,
        bbox=bbox, length_unit="kpc", unit_system="galactic"
        )
    
    left_edge = [-radius, -radius, -radius]
    right_edge = [radius, radius, radius]
    projection = data_source.box(left_edge, right_edge)
    
    generate_plots(plot_path, data_source, projection, 'gas', ('gas', 'density'), subhalo_ID, radius, axis, cmap)
    
    
    
    
    return()


def plot_stars(plot_path,
               subhalo_ID,
               snapshot,
               subhalo_stars_physical_grid,
               radius=15,
               cmap='arbre',
               axis='z'):
    
    bbox = np.array([[-radius, radius],
                     [-radius, radius],
                     [-radius, radius]])

    data_source = yt.load_uniform_grid(
        subhalo_stars_physical_grid,
        subhalo_stars_physical_grid[('gas', 'density')][0].shape,
        bbox=bbox, length_unit="kpc", unit_system="galactic"
        )
    
    left_edge = [-radius, -radius, -radius]
    right_edge = [radius, radius, radius]
    projection = data_source.box(left_edge, right_edge)
    
    generate_plots(plot_path, data_source, projection, 'stars', ('gas', 'density'), subhalo_ID, radius, axis, cmap)
    
    
    
    
    return()


def plot_dm(plot_path,
            subhalo_ID,
            snapshot,
            subhalo_dm_physical_grid,
            radius=15,
            cmap='arbre',
            axis='z'):
    
    bbox = np.array([[-radius, radius],
                     [-radius, radius],
                     [-radius, radius]])

    data_source = yt.load_uniform_grid(
        subhalo_dm_physical_grid,
        subhalo_dm_physical_grid[('gas', 'density')][0].shape,
        bbox=bbox, length_unit="kpc", unit_system="galactic"
        )
    
    left_edge = [-radius, -radius, -radius]
    right_edge = [radius, radius, radius]
    projection = data_source.box(left_edge, right_edge)
    
    generate_plots(plot_path, data_source, projection, 'dm', ('gas', 'density'), subhalo_ID, radius, axis, cmap)
    
    
    
    
    return()