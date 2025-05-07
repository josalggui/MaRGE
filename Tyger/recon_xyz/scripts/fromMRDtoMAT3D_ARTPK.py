import sys
import argparse
import mrd
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def export(input, output):
    with mrd.BinaryMrdReader(input) as r:
        header = r.read_header()
        for item in r.read_data():
            if not isinstance(item, mrd.StreamItem.ImageFloat):
                raise RuntimeError("Stream must contain only floating point images")

            img = item.value
            imgRecon = img.data
    # print(imgRecon.shape)  
    # img3D = imgRecon[0]
    # plt.figure(figsize = (5,8), dpi=240)
    # gs1 = gridspec.GridSpec(5,8)
    # gs1.update(wspace=0.020, hspace=0.020) 

    # for i in range(38):
    #     if i > img3D.shape[0]:
    #         break
    #     ax1 = plt.subplot(gs1[i])
    #     plt.axis('off')
    #     ax1.set_xticklabels([])
    #     ax1.set_yticklabels([])
    #     ax1.set_aspect('equal')
    #     imgAux = img3D[:,:,int(i)]
    #     ax1.imshow(np.abs(imgAux),cmap='gray')

    # plt.savefig('/home/tyger/tyger_repo_may/Tyger_MRIlab/resultART.png')

    rawData = sio.loadmat(output)
    rawData['imgReconTyger_ARTPK'] = imgRecon
    sio.savemat(output, rawData)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save recon in .mat file")
    parser.add_argument('-i', '--input', type=str, required=False,
                        help="Input file (default stdin)")
    parser.add_argument('-o', '--output', type=str, required=False,
                        help="Output filename .mat")
    # parser.set_defaults(
    #     input = '/home/tyger/tyger_repo_may/Tyger_MRIlab/reconART.bin',
    #     output= '/home/tyger/tyger_repo_may/brainIR.mat',
    # )
    args = parser.parse_args()

    input = open(args.input, "rb") if args.input is not None else sys.stdin.buffer

    export(input, args.output)
