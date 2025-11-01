import os

import numpy as np
import scipy as sp
from skimage.restoration import unwrap_phase as unwrap
from sklearn.preprocessing import PolynomialFeatures

from marge.configs import hw_config as hw
from marge.marge_utils.utils import run_ifft


def SPDS(raw_data_path=None):
    """
    Analyzes the data acquired from the SPDS (Single Point Double Shot) sequence to estimate the B₀ map,
    generate k-space and spatial domain images, and prepare the outputs for visualization.

    Parameters:
    -----------
    mode : str, optional
        Execution mode of the analysis. If set to 'Standalone', the results are plotted immediately
        after analysis (default: None).

    Outputs:
    --------
    - `output` (list): A list of dictionaries defining the data and parameters for visualization.
      Includes:
        - Spatial domain magnitude images for channels A and B.
        - B₀ field map.
        - k-space magnitude images for channels A and B.
    - Updates `self.mapVals` with intermediate results, including k-space, spatial images, and the
      B₀ field map.
    - If `mode == 'Standalone'`, plots the results.

    Notes:
    ------
    - Assumes that the k-space mask and orientation settings are correctly preconfigured.
    """

    if raw_data_path is None:
        return None

    # load .mat
    mat_data = sp.io.loadmat(raw_data_path)

    # Create new dictionary to save new outputs
    output_dict = {}

    # Print inputs
    keys = mat_data['input_keys']
    strings = mat_data['input_strings']
    string = ""
    print("****Inputs****")
    for ii, key in enumerate(keys):
        string = string + f"{str(strings[ii]).strip()}: {np.squeeze(mat_data[str(key).strip()])}, "
    print(string)
    print("****Outputs****")

    def zero_padding(data, order):
        original_shape = data.shape
        if len(original_shape) == 3:
            if original_shape[0] == 1:
                new_shape = (1, original_shape[1] * order, original_shape[2] * order)
            else:
                new_shape = tuple(dim * order for dim in original_shape)
        else:
            raise ValueError("Error of matrix shape")

        k_dataZP_a = np.zeros(new_shape, dtype=data.dtype)
        start_indices = tuple((new_dim - old_dim) // 2 for new_dim, old_dim in zip(new_shape, original_shape))
        end_indices = tuple(start + old_dim for start, old_dim in zip(start_indices, original_shape))
        if original_shape[0] == 1:
            k_dataZP_a[0, start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]] = data[0]
        else:
            k_dataZP_a[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1],
            start_indices[2]:end_indices[2]] = data

        return k_dataZP_a

    # Load data
    data_a = mat_data['data_decimated_a'][0]
    data_b = mat_data['data_decimated_b'][0]
    k_points = mat_data['k_cartesian']
    mask = mat_data['mask'][0]
    n_points = mat_data['nPoints'][0]
    interp_order = mat_data['interpOrder'][0][0]
    threshold_mask = mat_data['thresholdMask'][0][0]
    dead_time = mat_data['deadTime'][0] * 1e-6  # s
    fov = mat_data['fov'][0] * 1e-2  # m
    fitting_order = mat_data['fittingOrder'][0][0]

    # Delete the addRdPoints and last readout
    data_a = np.reshape(data_a, (-1, 1 + 2 * hw.addRdPoints))
    data_b = np.reshape(data_b, (-1, 1 + 2 * hw.addRdPoints))
    data_a = data_a[0:-1, hw.addRdPoints]
    data_b = data_b[0:-1, hw.addRdPoints]

    # Fill k_space
    k_data_a = np.zeros(np.size(k_points, 0), dtype=complex)
    k_data_b = np.zeros(np.size(k_points, 0), dtype=complex)
    jj = 0
    for ii in range(np.size(mask)):
        if mask[ii]:
            k_data_a[ii] = data_a[jj]
            k_data_b[ii] = data_b[jj]
            jj += 1

    # Get images
    k_data_aRaw = (np.reshape(k_data_a, (n_points[2], n_points[1], n_points[0])))
    k_data_bRaw = (np.reshape(k_data_b, (n_points[2], n_points[1], n_points[0])))
    k_data_a = zero_padding(k_data_aRaw, interp_order)
    k_data_b = zero_padding(k_data_bRaw, interp_order)

    i_data_a = run_ifft(k_data_a)
    i_data_b = run_ifft(k_data_b)
    output_dict['space_k_a'] = k_data_a
    output_dict['space_k_b'] = k_data_b
    output_dict['space_i_a'] = i_data_a
    output_dict['space_i_b'] = i_data_b

    # Plots in GUI
    if n_points[2] == 1:
        i_data_a = np.squeeze(i_data_a)
        i_data_b = np.squeeze(i_data_b)

        # Generate mask
        p_max = np.max(np.abs(i_data_a))
        mask = np.abs(i_data_a) < p_max * threshold_mask/100

        # Get phase
        RawPhase1 = np.angle(i_data_a)
        RawPhase1[mask] = 0
        RawPhase2 = np.angle(i_data_b)
        RawPhase2[mask] = 0

        i_phase_a = unwrap(RawPhase1)
        i_phase_b = unwrap(RawPhase2)

        # Get magnetic field
        b_field = ((i_phase_b - i_phase_a) / (2 * np.pi * hw.gammaB * (dead_time[1] - dead_time[0])))
        b_field[mask] = 0
        output_dict['b_field'] = b_field

        NX = n_points[0]*interp_order
        NY = n_points[1]*interp_order
        dx = fov[0] / NX
        dy = fov[1] / NY

        # Here we define the grid of the full FOV and select the indexs where B0 is no null
        ii, jj = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
        condition = b_field != 0
        ii = ii[condition]
        jj = jj[condition]

        # Here we define the coordinates of the FOV where the B0 is no null
        x = (-(NX - 1) / 2 + ii) * dx
        y = (-(NY - 1) / 2 + jj) * dy

        # Store in values the B0 value in the indexs that accomplishes B0 different of 0
        values = b_field[condition]

        # Save in mapList all the {╥x,y,z,B0} data where B0 is no null
        mapList = np.column_stack((x, y, values))
        output_dict['mapList'] = mapList

        # And now we proceed with the fitting
        x_fit = mapList[:, 0]
        y_fit = mapList[:, 1]
        B_fit = mapList[:, 2]
        poly = PolynomialFeatures(fitting_order)
        coords = np.vstack((x_fit, y_fit)).T  # Combinamos x, y como entrada para PolynomialFeatures
        X_poly = poly.fit_transform(coords)  # Genera todas las combinaciones polinómicas hasta grado 4
        coeffs, _, _, _ = np.linalg.lstsq(X_poly, B_fit, rcond=None)
        terms = terms = poly.powers_
        polynomial_expression = ""
        for i, coeff in enumerate(coeffs):
            if coeff != 0:  # Ignore null coefficients
                powers = terms[i]  # Powers (x**i, y**j)
                term = f"{coeff}"
                if any(powers):
                    if powers[0] > 0:
                        term += f"*(x**{powers[0]})"
                    if powers[1] > 0:
                        term += f"*(y**{powers[1]})"
                polynomial_expression += f" + {term}" if coeff > 0 and i > 0 else f" {term}"

        print("B0 fitting:")
        print(polynomial_expression)

        result1 = {}
        result1['widget'] = 'image'
        result1['data'] = np.real(b_field.reshape(1, NX, NY))
        result1['xLabel'] = "xx"
        result1['yLabel'] = "xx"
        result1['title'] = "B0 field"
        result1['row'] = 0
        result1['col'] = 3

        result4 = {}
        result4['widget'] = 'image'
        result4['data'] = np.real(RawPhase1.reshape(1, NX, NY))
        result4['xLabel'] = "xx"
        result4['yLabel'] = "xx"
        result4['title'] = "Raw Phase Image Td1"
        result4['row'] = 0
        result4['col'] = 1

        result5 = {}
        result5['widget'] = 'image'
        result5['data'] = np.real(RawPhase2.reshape(1, NX, NY))
        result5['xLabel'] = "xx"
        result5['yLabel'] = "xx"
        result5['title'] = "Raw Phase Image Td2"
        result5['row'] = 1
        result5['col'] = 1

        result2 = {}
        result2['widget'] = 'image'
        result2['data'] = np.real(i_phase_a.reshape(1, NX, NY))
        result2['xLabel'] = "xx"
        result2['yLabel'] = "xx"
        result2['title'] = "Unwrapped Phase Image Td1"
        result2['row'] = 0
        result2['col'] = 2

        result3 = {}
        result3['widget'] = 'image'
        result3['data'] = np.real(i_phase_b.reshape(1, NX, NY))
        result3['xLabel'] = "xx"
        result3['yLabel'] = "xx"
        result3['title'] = "Unwrapped Phase Image Td2"
        result3['row'] = 1
        result3['col'] = 2

        result6 = {}
        result6['widget'] = 'image'
        result6['data'] = np.abs(i_data_a.reshape(1, NX, NY))
        result6['xLabel'] = "xx"
        result6['yLabel'] = "xx"
        result6['title'] = "Raw Abs Image Td1"
        result6['row'] = 0
        result6['col'] = 0

        result7 = {}
        result7['widget'] = 'image'
        result7['data'] = np.abs(i_data_b.reshape(1, NX, NY))
        result7['xLabel'] = "xx"
        result7['yLabel'] = "xx"
        result7['title'] = "Raw Abs Image Td2"
        result7['row'] = 1
        result7['col'] = 0

        outputs = [result1, result2, result3, result4, result5, result6, result7]

    if n_points[0] > 1 and n_points[1] > 1 and n_points[2] > 1:
        # Generate mask
        p_max = np.max(np.abs(i_data_a))
        mask = np.abs(i_data_a) < p_max * threshold_mask/100

        # Get phase
        RawPhase1 = np.angle(i_data_a)
        RawPhase1[mask] = 0
        RawPhase2 = np.angle(i_data_b)
        RawPhase2[mask] = 0

        i_phase_a = unwrap(RawPhase1)
        i_phase_b = unwrap(RawPhase2)

        # Get magnetic field
        b_field = -(i_phase_b - i_phase_a) / (2 * np.pi * hw.gammaB * (dead_time[1] - dead_time[0]))
        b_field[mask] = 0
        output_dict['b_field'] = b_field
        B0mapReorganized = np.flip(np.flip(np.flip(np.transpose(b_field, (2, 1, 0)), axis=0), axis=1), axis=2)
        output_dict['B0mapReorganized'] = B0mapReorganized

        NX = n_points[0] * interp_order
        NY = n_points[1] * interp_order
        NZ = n_points[2] * interp_order
        dx = fov[0] / NX
        dy = fov[1] / NY
        dz = fov[2] / NZ

        mapList = []
        cont = 0

        for ii in range(NX):
            for jj in range(NY):
                for kk in range(NZ):
                    if B0mapReorganized[ii, jj, kk] != 0:
                        z_coord = (-(NZ - 1) / 2 + kk) * dz
                        y_coord = (-(NY - 1) / 2 + jj) * dy
                        x_coord = (-(NX - 1) / 2 + ii) * dx
                        value = B0mapReorganized[ii, jj, kk]

                        mapList.append([x_coord, y_coord, z_coord, value])
                        cont += 1

        mapList = np.array(mapList)
        output_dict['mapList'] = mapList

        # And now we proceed with the fitting
        x_fit = mapList[:, 0]
        y_fit = mapList[:, 1]
        z_fit = mapList[:, 2]
        B_fit = mapList[:, 3]
        poly = PolynomialFeatures(fitting_order)
        coords = np.vstack((x_fit, y_fit, z_fit)).T
        X_poly = poly.fit_transform(coords)
        coeffs, _, _, _ = np.linalg.lstsq(X_poly, B_fit, rcond=None)
        terms = poly.powers_
        polynomial_expression = ""
        polynomial_expressionGUI = ""
        for i, coeff in enumerate(coeffs):
            if coeff != 0:  # Ignore null coefficients
                powers = terms[i]  # Powers (x**i, y**j, z**k)
                term = f"{coeff}"
                termGUI = f"{coeff}"
                if any(powers):
                    if powers[2] > 0:
                        term += f"*(z**{powers[2]})"
                        termGUI += f"*(z^{powers[2]})"
                    if powers[1] > 0:
                        term += f"*(y**{powers[1]})"
                        termGUI += f"*(y^{powers[1]})"
                    if powers[0] > 0:
                        term += f"*(x**{powers[0]})"
                        termGUI += f"*(x^{powers[0]})"
                polynomial_expression += f" + {term}" if coeff > 0 and i > 0 else f" {term}"
                polynomial_expressionGUI += f" + {termGUI}" if coeff > 0 and i > 0 else f" {termGUI}"

        print("B0 fitting:")
        print(polynomial_expressionGUI)

        result1 = {}
        result1['widget'] = 'image'
        result1['data'] = np.real(b_field)
        result1['xLabel'] = "xx"
        result1['yLabel'] = "xx"
        result1['title'] = "B0 field"
        result1['row'] = 0
        result1['col'] = 3

        result4 = {}
        result4['widget'] = 'image'
        result4['data'] = np.real(RawPhase1)
        result4['xLabel'] = "xx"
        result4['yLabel'] = "xx"
        result4['title'] = "Raw Phase Image Td1"
        result4['row'] = 0
        result4['col'] = 1

        result5 = {}
        result5['widget'] = 'image'
        result5['data'] = np.real(RawPhase2)
        result5['xLabel'] = "xx"
        result5['yLabel'] = "xx"
        result5['title'] = "Raw Phase Image Td2"
        result5['row'] = 1
        result5['col'] = 1

        result2 = {}
        result2['widget'] = 'image'
        result2['data'] = np.real(i_phase_a)
        result2['xLabel'] = "xx"
        result2['yLabel'] = "xx"
        result2['title'] = "Unwrapped Phase Image Td1"
        result2['row'] = 0
        result2['col'] = 2

        result3 = {}
        result3['widget'] = 'image'
        result3['data'] = np.real(i_phase_b)
        result3['xLabel'] = "xx"
        result3['yLabel'] = "xx"
        result3['title'] = "Unwrapped Phase Image Td2"
        result3['row'] = 1
        result3['col'] = 2

        result6 = {}
        result6['widget'] = 'image'
        result6['data'] = np.abs(i_data_a)
        result6['xLabel'] = "xx"
        result6['yLabel'] = "xx"
        result6['title'] = "Raw Abs Image Td1"
        result6['row'] = 0
        result6['col'] = 0

        result7 = {}
        result7['widget'] = 'image'
        result7['data'] = np.abs(i_data_b)
        result7['xLabel'] = "xx"
        result7['yLabel'] = "xx"
        result7['title'] = "Raw Abs Image Td2"
        result7['row'] = 1
        result7['col'] = 0

        outputs = [result1, result2, result3, result4, result5, result6, result7]

    # Save polynomial_expression into mapVals
    output_dict['polynomial_expression'] = polynomial_expression

    return output_dict, outputs

if __name__ == '__main__':
    SPDS(raw_data_path='raw_data.2025.08.03.07.54.03.786.mat')