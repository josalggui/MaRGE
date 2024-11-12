import sys
import argparse
import mrd
import scipy.io as sio
import numpy as np

def export(input, output):
    with mrd.BinaryMrdReader(input) as r:
        header = r.read_header()
        for item in r.read_data():
            if not isinstance(item, mrd.StreamItem.ImageFloat):
                raise RuntimeError("Stream must contain only floating point images")

            img = item.value
            imgRecon = img.data
            
    rawData = sio.loadmat(output)
    rawData['imgReconTyger2D'] = imgRecon
    sio.savemat(output, rawData)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save recon in .mat file")
    parser.add_argument('-i', '--input', type=str, required=False,
                        help="Input file (default stdin)")
    parser.add_argument('-o', '--output', type=str, required=False,
                        help="Output filename .mat")
    # parser.set_defaults(
    #     input = 'reconBartCS.bin',
    #     output= '/home/teresa/Documentos/Tyger/pythonMRDi3m/customDockerTyger/scripts/knee44L.mat',
    # )
    args = parser.parse_args()

    input = open(args.input, "rb") if args.input is not None else sys.stdin.buffer

    export(input, args.output)
