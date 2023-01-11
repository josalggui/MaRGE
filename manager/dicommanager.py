import numpy as np
from pydicom import dcmread
from pydicom.data import get_testdata_file
import datetime
import pytz
from scipy import ndimage


class DICOMImage:
    def __init__(self):
        # Cargar el archivo de prueba de DICOM
        filename = get_testdata_file("MR_small.dcm")
        self.ds = dcmread(filename)

    def image2Dicom(self, x):
        for key in x.keys():
            setattr(self.ds, key, x[key])
        current_time = datetime.datetime.now().astimezone(datetime.timezone.utc).astimezone(pytz.timezone("Europe/Madrid"))
        self.ds.AcquisitionTime = current_time.strftime("%H%M%S.%f")

    def save(self, filename):
        self.ds.save_as(filename)

    def __str__(self):
        return f"Imagen DICOM con nombre del paciente {self.ds.PatientName} y ID {self.ds.PatientID}"


if __name__ == '__main__':
    # Crear una lista para almacenar las imágenes 2D
    images_2d = []

    # Crear una nueva imagen DICOM
    x = {}
    x["PatientName"] = "Pedro el Cruel"
    x["PatientID"] = "123456"

    # Configurar los detalles de la imagen
    rows = 100
    columns = 100
    frames = 10

    # Crear una matriz 2D con valores de ejemplo
    arr = np.zeros((frames, columns, rows), dtype=np.int16)
    arr[0, 0:20, 0:20] = 50
    arr[4, 0:40, 0:40] = 100
    x["Columns"] = columns
    x["Rows"] = rows
    x["NumberOfFrames"] = frames
    x["PixelData"] = arr.tobytes()

    # Crear una nueva imagen DICOM y configurar los detalles de la imagen
    dicom_image = DICOMImage()
    dicom_image.image2Dicom(x)

    dicom_image.save("C:\Users\TuNombre\Escritorio\Ejemplo 7 3D.dcm")

