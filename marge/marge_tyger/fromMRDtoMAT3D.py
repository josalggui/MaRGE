import sys
import argparse
import mrd
import scipy.io as sio
import numpy as np


def export(input, output, out_field):

    with mrd.BinaryMrdReader(input) as r:
        header = r.read_header()

        # Read axesOrientation: maps acq dims (rd=0, ph=1, sl=2) to spatial axes (x=0, y=1, z=2)
        axesOrientation = [0, 1, 2]  # default: rd=x, ph=y, sl=z
        if header.user_parameters:
            for param in header.user_parameters.user_parameter_string:
                if param.name == 'axesOrientation':
                    axesOrientation = [int(v) for v in param.value.split(',')]
                    break

        for item in r.read_data():
            if not isinstance(item, mrd.StreamItem.ImageFloat):
                raise RuntimeError("Stream must contain only floating point images")

            img = item.value
            # img.data is (ch, x, y, z) in physical space.
            # axesOrientation[k] = spatial axis (0=x,1=y,2=z) of acquisition dim k (0=rd,1=ph,2=sl).
            # Reorder to MaRGE format (ch, sl, ph, rd):
            perm = (0, 1 + axesOrientation[2], 1 + axesOrientation[1], 1 + axesOrientation[0])
            imgRecon = np.transpose(img.data, perm)

    rawData = sio.loadmat(output)
    rawData[out_field] = imgRecon
    sio.savemat(output, rawData)
    return imgRecon
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save recon in .mat file")
    parser.add_argument('-i', '--input', type=str, required=False, help="Input file (default stdin)")
    parser.add_argument('-o', '--output', type=str, required=False, help="Output filename .mat")
    parser.add_argument('-of', '--out_field', type=str, required=False, help="Recon img name field")
    
    # parser.set_defaults(
    #     input = '/home/teresa/marcos_tyger/Brain_Images/output.bin',
    #     output= '/home/teresa/marcos_tyger/Brain_Images/brainIR.mat',
    #     out_field = 'tyger_test'
    # )
    
    args = parser.parse_args()

    input = open(args.input, "rb") if args.input is not None else sys.stdin.buffer

    export(input, args.output, args.out_field)
