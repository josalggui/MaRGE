import sys
import argparse
import mrd
import scipy.io as sio


def export(input, output, out_field):
    
    with mrd.BinaryMrdReader(input) as r:
        header = r.read_header()
        for item in r.read_data():
            if not isinstance(item, mrd.StreamItem.ImageFloat):
                raise RuntimeError("Stream must contain only floating point images")

            img = item.value
            imgRecon = img.data
            
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
