import pandas as pd
import numpy as np

def get_surf_mesh(geometry):
    
    surf_dict = { }
    surf_dict['spline_0'] = 'Meshpkg/Initialize/training_files/spline_#100_surf_2d.txt'
    surf_dict['spline_1'] = 'Meshpkg/Initialize/training_files/spline_1_#34_surf_2d.txt'
    surf_dict['spline_1_1'] = 'Meshpkg/Initialize/training_files/spline_1_1_#16_surf_2d.txt'
    surf_dict['spline_2'] = 'Meshpkg/Initialize/training_files/spline_2_#38_surf_2d.txt'
    surf_dict['spline_3'] = 'Meshpkg/Initialize/training_files/spline_3_#40_surf_2d.txt'
    surf_dict['airfoil_sharp'] = 'Meshpkg/Initialize/training_files/airfoil0012_sharp_#100_surf_2d.txt'

    surf_df = pd.read_csv(surf_dict[geometry], sep = " ", header=None) # sep = "\t"," ", "  "

    return np.array(surf_df)