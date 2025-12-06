import sys
import argparse
import mrd
import scipy.io as sio
import numpy as np

def export(input, output, out_field, out_field_k):
    images = []
    with mrd.BinaryMrdReader(input) as reader:
        header = reader.read_header()
        assert header is not None, "No header found in reconstructed file"

        for item in reader.read_data():
            if isinstance(item, mrd.StreamItem.ImageFloat):
                images.append(item.value)
    
    imgRecon = images[0].data        
    rawData = sio.loadmat(output)
    rawData[out_field] = imgRecon[0]
    kSpace3D_den = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(imgRecon[0])))
    rawData[out_field_k] = kSpace3D_den
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
