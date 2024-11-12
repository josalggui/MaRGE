import scipy.io as sio
import pydicom

file_path = 'Vol38s18/RAREprotocols_T1_SAG_Right.2023.07.06.11.33.29.920.mat'
highfield_path = 'Vol38s18/38HF.dcm'

ds = pydicom.dcmread(highfield_path)
img_3t = ds.pixel_array 

rawData = sio.loadmat(file_path)
rawData['highfield_img'] = img_3t
rawData['imgReconTyger2D_Red'] = img_3t * 0
sio.savemat(file_path, rawData)

rawData = sio.loadmat(file_path)
print(rawData['highfield_img'].shape)

rawData = sio.loadmat(file_path)
print(rawData['highfield_img'].shape)