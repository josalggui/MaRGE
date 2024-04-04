import numpy as np
from pydicom import dcmread
from pydicom.data import get_testdata_file
import datetime


class DICOMImage:
    def __init__(self, path=None):
        # Cargar el archivo de prueba de DICOM
        self.meta_data = {}
        if path is None:
            path = get_testdata_file("MR_small.dcm")
        try:
            self.ds = dcmread(path)
        except:
            self.ds = dcmread(get_testdata_file("MR_small.dcm"))

    def image2Dicom(self):
        for key in self.meta_data.keys():
            setattr(self.ds, key, self.meta_data[key])

    def save(self, filename):
        self.ds.save_as(filename)

    def __str__(self):
        return f"Imagen DICOM con nombre del paciente {self.ds.PatientName} y ID {self.ds.PatientID}"


if __name__ == '__main__':
    #### Random image
    name = 'Random'
    nRd = 90
    nPh = 90
    allSlices = 5
    arr = np.zeros((allSlices, nPh, nRd), dtype=np.int16)
    arr[0, 0:20, 0:20] = 50
    arr[4, 0:40, 0:40] = 100
    repetitionTime = 200
    echoTime = 20
    ETL = 5

    #### Crear una nueva imagen DICOM
    # Image data
    dicom_image = DICOMImage()
    dicom_image.meta_data["PixelData"] = arr.tobytes()

    # Date and time
    current_time = datetime.datetime.now()
    dicom_image.meta_data["StudyDate"] = current_time.strftime("%Y%m%d")
    dicom_image.meta_data["StudyTime"] = current_time.strftime("%H%M%S")

    # Sequence parameters
    dicom_image.meta_data["Columns"] = nRd
    dicom_image.meta_data["Rows"] = nPh
    dicom_image.meta_data["NumberOfSlices"] = allSlices
    dicom_image.meta_data["RepetitionTime"] = repetitionTime
    dicom_image.meta_data["EchoTime"] = echoTime
    dicom_image.meta_data["EchoTrainLength"] = ETL

    # More DICOM tags
    # dicom_image.meta_data["SeriesNumber"] = 1
    dicom_image.meta_data["PatientName"] = " "
    dicom_image.meta_data["PatientID"] = name
    dicom_image.meta_data["PatientSex"] = " "
    dicom_image.meta_data["StudyID"] = "KneeProtocols"
    dicom_image.meta_data["InstitutionName"] = "PhysioMRI"
    dicom_image.meta_data["ImageComments"] = " "
    SOPInstanceUID = name
    dicom_image.meta_data["SOPInstanceUID"] = SOPInstanceUID

    #### Create DICOM file
    #### DICOM 3.0 ####
    dicom_image.meta_data["NumberOfFrames"] = allSlices
    dicom_image.image2Dicom()
    nameDcmFile = name + ".dcm"
    dicom_image.save("C:/Users/Physio MRI/Desktop/Dicom prueva/Ejemplo_1.dcm")
