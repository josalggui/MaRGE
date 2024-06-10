def main(plane, read_grad, phase_grad, x_shim, y_shim, z_shim, x_cal, y_cal, z_cal):
    # Read - n1,n2,n9 (shim)
    # Phase - n3,n4,n8 (shim)
    # Other - n5,n6(shim)
    # Shims are stored in mT/m not T/m
    x_shim = x_shim / 1000
    y_shim = y_shim / 1000
    z_shim = z_shim / 1000
    if plane == "xy":
        n1 = 3  # x (read)`
        n3 = 2  # y (phase)
        n5 = 1  # z (shim)
        n2 = (x_shim + read_grad) * x_cal
        n4 = (phase_grad + y_shim) * y_cal
        n6 = z_shim * z_cal
        n8 = y_shim * y_cal
        n9 = x_shim * x_cal
    elif plane == "yx":
        n1 = 2  # y
        n3 = 3  # x
        n5 = 1  # z
        n2 = (y_shim + read_grad) * y_cal
        n4 = (phase_grad + x_shim) * x_cal
        n6 = z_shim * z_cal
        n8 = x_shim * x_cal
        n9 = y_shim * y_cal
    elif plane == "yz":
        n1 = 2  # y
        n3 = 1  # z
        n5 = 3  # x
        n2 = (y_shim + read_grad) * y_cal
        n4 = (phase_grad + z_shim) * z_cal
        n6 = x_shim * x_cal
        n8 = z_shim * z_cal
        n9 = y_shim * y_cal
    elif plane == "zy":
        n1 = 1  # z (read)
        n3 = 2  # y (phase)
        n5 = 3  # x (shim)
        n2 = (z_shim + read_grad) * z_cal
        n4 = (phase_grad + y_shim) * y_cal
        n6 = x_shim * x_cal
        n8 = y_shim * y_cal
        n9 = z_shim * z_cal
    elif plane == "xz":
        n1 = 3  # x
        n3 = 1  # z
        n5 = 2  # y
        n2 = (x_shim + read_grad) * x_cal

        n4 = (phase_grad + z_shim) * z_cal
        n6 = y_shim * y_cal
        n8 = z_shim * z_cal
        n9 = x_shim * x_cal
    elif plane == "zx":
        n1 = 1  # x
        n3 = 3  # z
        n5 = 2  # y
        n2 = (z_shim + read_grad) * z_cal
        n4 = (phase_grad + x_shim) * x_cal
        n6 = y_shim * y_cal
        n8 = x_shim * x_cal
        n9 = z_shim * z_cal
    else:
        raise Exception("Invalid plane")

    return n1, n2, n3, n4, n5, n6, n8, n9
