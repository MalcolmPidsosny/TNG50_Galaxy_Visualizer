# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:41:30 2026

@author: MalcolmP
"""


import numpy as np
import yt


def generate_plots(plot_path, data_source, field, subhalo_ID, radius, axis, cmap):
    
    prj = yt.ProjectionPlot(data_source, "z", field=field, center="c", width=(20, "kpc"), data_source=data_source)
    prj.set_cmap(field=(field), cmap=cmap)
    prj.annotate_title(f'Subhalo ID: {subhalo_ID}; {field}')
    prj.save(f"{plot_path}/{subhalo_ID}_{field}")
    
    return()


def plot_gas(plot_path, subhalo_ID, snapshot, subhalo_gas_physical_grid, radius=15, cmap='arbre', axis='z'):
    
    bbox = np.array([[-radius, radius],
                     [-radius, radius],
                     [-radius, radius]])

    data_source = yt.load_uniform_grid(
        subhalo_gas_physical_grid, subhalo_gas_physical_grid['gas', 'density'].shape, bbox=bbox, length_unit="kpc"
        )
    
    left_edge = [-radius, -radius, -radius]
    right_edge = [radius, radius, radius]
    data_source = data_source.box(left_edge, right_edge)
    
    for field in data_source.field_list:
        generate_plots(plot_path, data_source, field, subhalo_ID, radius, axis, cmap)
    
    
    print(f'Field List: {data_source.field_list}')
    
    return()


def plot_stars(plot_path, subhalo_ID, snapshot, subhalo_stars_physical_grid, radius=15, cmap='arbre', axis='z'):
    
    bbox = np.array([[-radius, radius],
                     [-radius, radius],
                     [-radius, radius]])

    data_source = yt.load_uniform_grid(
        subhalo_stars_physical_grid,
        subhalo_stars_physical_grid['stars', 'density'].shape,
        bbox=bbox, length_unit="kpc"
        )
    
    left_edge = [-radius, -radius, -radius]
    right_edge = [radius, radius, radius]
    data_source = data_source.box(left_edge, right_edge)
    
    for field in data_source.field_list:
        generate_plots(plot_path, data_source, field, subhalo_ID, radius, axis, cmap)
    
    
    print(f'Field List: {data_source.field_list}')
    
    return()


def plot_dm(plot_path, subhalo_ID, snapshot, subhalo_dm_physical_grid, radius=15, cmap='arbre', axis='z'):
    
    bbox = np.array([[-radius, radius],
                     [-radius, radius],
                     [-radius, radius]])

    data_source = yt.load_uniform_grid(
        subhalo_dm_physical_grid, subhalo_dm_physical_grid['dm', 'density'].shape, bbox=bbox, length_unit="kpc"
        )
    
    left_edge = [-radius, -radius, -radius]
    right_edge = [radius, radius, radius]
    data_source = data_source.box(left_edge, right_edge)
    
    for field in data_source.field_list:
        generate_plots(plot_path, data_source, field, subhalo_ID, radius, axis, cmap)
    
    
    print(f'Field List: {data_source.field_list}')
    
    return()