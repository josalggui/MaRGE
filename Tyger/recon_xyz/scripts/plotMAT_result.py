import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

input = '/home/teresa/marcos_tyger/Tyger_MRIlab/recon_xyz/scripts/brainIR.mat'
rawData = sio.loadmat(input)
img3D = rawData['imgReconTyger'][0]
print(img3D.shape)
# print(np.max(img3D))

plt.figure(figsize = (5,8), dpi=240)
gs1 = gridspec.GridSpec(5,8)
gs1.update(wspace=0.020, hspace=0.020) # set the spacing between axes.

for i in range(38):
    if i > img3D.shape[0]:
        break
    ax1 = plt.subplot(gs1[i])
    plt.axis('off')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_aspect('equal')
    imgAux = img3D[:,:,int(i)]
    ax1.imshow(imgAux,cmap='gray')

plt.show()
# plt.savefig('/home/tyger/tyger_repo_may/Tyger_MRIlab/resultTEST.png')