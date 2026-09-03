# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:47:48 2026

@author: josit
"""

import sys

sys.path.insert(
    0,
    '/home/malcolmp/Notebooks/TNG50_Galaxy_Visualizer/src'
    )


from main import run_tng50_visualizer


config_dir = '/home/malcolmp/Notebooks/TNG50_Galaxy_Visualizer/configs'


if __name__ == "__main__":
    
    config_file = f'{config_dir}/{sys.argv[1]}'
    
    run_tng50_visualizer(config_file)