import sys
import argparse
import numpy as np
from typing import BinaryIO, Iterable, Union
import mrd
import matplotlib.pyplot as plt
# from bart_marcos import bart_marcos2D
import cupy as cp 

def acquisition_reader(input: Iterable[mrd.StreamItem]) -> Iterable[mrd.Acquisition]:
    for item in input:
        if not isinstance(item, mrd.StreamItem.Acquisition):
            # Skip non-acquisition items
            continue
        if item.value.flags & mrd.AcquisitionFlags.IS_NOISE_MEASUREMENT:
            # Currently ignoring noise scans
            continue
        yield item.value

def stream_item_sink(input: Iterable[Union[mrd.Acquisition, mrd.Image[np.float32]]]) -> Iterable[mrd.StreamItem]:
    for item in input:
        if isinstance(item, mrd.Acquisition):
            yield mrd.StreamItem.Acquisition(item)
        elif isinstance(item, mrd.Image) and item.data.dtype == np.float32:
            yield mrd.StreamItem.ImageFloat(item)
        else:
            raise ValueError("Unknown item type")

def mrdRecon(reconMode: str, artMode:str,
              head: mrd.Header, input: Iterable[mrd.Acquisition]) -> Iterable[mrd.Image[np.float32]]:
    
    if head.user_parameters and head.user_parameters.user_parameter_double:
        for param in head.user_parameters.user_parameter_double:
            if param.name == "readout_gradient_intensity":
                rdGradAmplitude = param.value
                break
    
    enc = head.encoding[0]

    # Matrix size
    if enc.encoded_space and enc.recon_space and enc.encoded_space.matrix_size and enc.recon_space.matrix_size:
        eNx = enc.encoded_space.matrix_size.x
        eNy = enc.encoded_space.matrix_size.y
        eNz = enc.encoded_space.matrix_size.z
        rNx = enc.recon_space.matrix_size.x
        rNy = enc.recon_space.matrix_size.y
        rNz = enc.recon_space.matrix_size.z
    else:
        raise Exception('Required encoding information not found in header')

    # Field of view
    if enc.recon_space and enc.recon_space.field_of_view_mm:
        rFOVx = enc.recon_space.field_of_view_mm.x*1e-3
        rFOVy = enc.recon_space.field_of_view_mm.y*1e-3
        rFOVz = enc.recon_space.field_of_view_mm.z*1e-3 if enc.recon_space.field_of_view_mm.z else 1
    else:
        raise Exception('Required field of view information not found in header')

    kSpace_buffer = None
    kx_buffer = None
    ky_buffer = None
    kz_buffer = None
    times_buffer = None
    axesOrientation = None

    def produce_image(img: np.ndarray) -> Iterable[mrd.Image[np.float32]]:
        mrd_image = mrd.Image[np.float32](image_type=mrd.ImageType.MAGNITUDE, data=img)
        yield mrd_image
    
    # kSpace_buffer = np.empty((0,))
    kSpace_buffer = []
    kx_buffer = []
    ky_buffer = []
    kz_buffer = []
    times_buffer = []

    for acq in input:
        if axesOrientation == None:
            axesOrientation = acq.channel_order

        k1 = acq.idx.kspace_encode_step_1 if acq.idx.kspace_encode_step_1 is not None else 0
        k2 = acq.idx.kspace_encode_step_2 if acq.idx.kspace_encode_step_2 is not None else 0

        # # kSpace_buffer = np.concatenate((kSpace_buffer, acq.data[0]), axis = 0) # Mucho más lento!
        kSpace_buffer.append(acq.data[0])
        kx_buffer.append(acq.trajectory[0,:])
        ky_buffer.append(acq.trajectory[1,:])
        kz_buffer.append(acq.trajectory[2,:])
        times_buffer.append(acq.trajectory[3,:])

    # kSpace_buffer = np.array(kSpace_buffer)
    kSpace_buffer = np.reshape(kSpace_buffer, -1, order='C')
    kx_buffer = np.reshape(kx_buffer, -1, order='C')
    ky_buffer = np.reshape(ky_buffer, -1, order='C')
    kz_buffer= np.reshape(kz_buffer, -1, order='C')
    times_buffer = np.reshape(times_buffer, -1, order='C')

    def pythonfft(kSpace):        
        img = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kSpace[:, :, :])))
        img = np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
        img = np.abs(img).astype(np.float32)
        return img
    
    def pythonART():
        gammabar = 42.57747892*1e6

        signal = kSpace_buffer
        kx = 2*np.pi*kx_buffer
        ky = 2*np.pi*ky_buffer
        kz = 2*np.pi*kz_buffer

        x_vals = np.linspace(-rFOVx / 2 + rFOVx / (2 * rNx) , rFOVx / 2 + rFOVx / (2 * rNx), rNx)
        y_vals = np.linspace(-rFOVy / 2 + rFOVy / (2 * rNy) , rFOVy / 2 + rFOVy / (2 * rNy) , rNy)
        z_vals = np.linspace(-rFOVz / 2 + rFOVz / (2 * rNz) , rFOVz / 2 + rFOVz / (2 * rNz), rNz)
        x, y, z = np.meshgrid(x_vals, y_vals, z_vals,indexing='ij')
        x = x.flatten(order='F') 
        y = y.flatten(order='F')
        z = z.flatten(order='F')

        def boFit(x, y, z):
            return(1.7958916691214196e-06 + 8.780172101062597e-05*(x**1) + 5.1529384399101336e-05*(y**1) + 5.496867688951115e-06*(z**1) 
            + 0.003470212745476351*(x**2) + 0.008922537970838425*(y**1)*(x**1) -0.002658533047887016*(z**1)*(x**1) -0.004877821311734011*(y**2) 
            + 0.011312469067932374*(z**1)*(y**1) -0.004254244444363751*(z**2) + 0.011596056164130399*(x**3) -0.10099062035111861*(y**1)*(x**2) 
            + 0.23066792475669826*(z**1)*(x**2) + 0.06522447284230339*(y**2)*(x**1) -0.09268890836211033*(z**1)*(y**1)*(x**1) -0.2646230382632592*(z**2)*(x**1) 
            + 0.03732888960934208*(y**3) -0.07965855164161945*(z**1)*(y**2) -0.0037222412970749352*(z**2)*(y**1) -0.0990178278290059*(z**3) -0.22386940676842962*(x**4) 
            + 0.4515742630482076*(y**1)*(x**3) -2.427971146354535*(z**1)*(x**3) + 0.8453349553024235*(y**2)*(x**2) + 0.2165221983624357*(z**1)*(y**1)*(x**2) 
            -0.7615407920517288*(z**2)*(x**2) -0.0906496147583209*(y**3)*(x**1) + 1.2421193883567103*(z**1)*(y**2)*(x**1) -1.2839843925917132*(z**2)*(y**1)*(x**1) 
            + 0.8144885330147165*(z**3)*(x**1) + 0.258365771390757*(y**4) -0.5281918562492409*(z**1)*(y**3) + 0.009060285917140476*(z**2)*(y**2) 
            -0.5552447960655633*(z**3)*(y**1) + 0.4740721467059541*(z**4) + 1.5422222012964693*(x**5) + 3.9561465059427485*(y**1)*(x**4) -28.516667829569244*(z**1)*(x**4) 
            -2.8957480824348707*(y**2)*(x**3) -1.3249401492772208*(z**1)*(y**1)*(x**3) + 16.110941965800595*(z**2)*(x**3) + 7.86723677725079*(y**3)*(x**2) 
            -25.429379131759436*(z**1)*(y**2)*(x**2) -20.220814228485104*(z**2)*(y**1)*(x**2) + 2.456315486745019*(z**3)*(x**2) + 1.4570480857538963*(y**4)*(x**1) 
            + 12.843202531324176*(z**1)*(y**3)*(x**1) + 1.5852877967582848*(z**2)*(y**2)*(x**1) + 6.105833933761557*(z**3)*(y**1)*(x**1) + 13.246906094350901*(z**4)*(x**1)
            -3.3671095590823654*(y**5) + 7.685302668620389*(z**1)*(y**4) + 8.679645703874112*(z**2)*(y**3) + 6.056476848223719*(z**3)*(y**2) -4.20548889854099*(z**4)*(y**1)
            + 7.385257017237823*(z**5))

        dBo = boFit(-x, -y, -z)
        
        rho = np.reshape(np.zeros((rNx*rNy*rNz), dtype=complex), (-1, 1))
        rho = rho[:,0]
        lbda = 1/float(1)
        n_iter = int(1)
        index = np.arange(len(rho))

        kx_gpu = cp.asarray(kx)
        ky_gpu = cp.asarray(ky)
        kz_gpu = cp.asarray(kz)
        sx_gpu = cp.asarray(x)
        sy_gpu = cp.asarray(y)
        sz_gpu = cp.asarray(z)
        signal_gpu = cp.asarray(signal)
        index_gpu = cp.asarray(index)
        rho_gpu = cp.asarray(rho)
        dBo_gpu = cp.asarray(dBo)
        timeVec = cp.asarray(times_buffer)

        def artPK(kx, ky, kz, x, y, z, s, rho, lbda, n_iter, index, times, dBo):
                    n = 0
                    n_samples = len(s)
                    m = 0
                    for iteration in range(n_iter):
                        # cp.random.shuffle(index)
                        for jj in range(n_samples):
                            ii = index[jj]
                            x0 = cp.exp(-1j *  (kx[ii] * x + ky[ii] * y + kz[ii] * z + 2 * np.pi * gammabar * times[ii]*dBo))
                            x1 =  s[ii]-(x0.T @ rho)
                            # x2 = x1 * cp.conj(x0) / (cp.conj(x0.T) @ x0)
                            x2 = x1 * cp.conj(x0) / (rNx*rNy*rNz)
                            d_rho = lbda * x2
                            rho += d_rho
                            n += 1
                            if n / n_samples > 0.01:
                                m += 1
                                n = 0
                                # print("ART iteration %i: %i %%" % (iteration + 1, m))

                    return rho
        
        def art(kx, ky, kz, x, y, z, s, rho, lbda, n_iter, index, times, dBo):
                    n = 0
                    n_samples = len(s)
                    m = 0
                    for iteration in range(n_iter):
                        # cp.random.shuffle(index)
                        for jj in range(n_samples):
                            ii = index[jj]
                            x0 = cp.exp(-1j *  (kx[ii] * x + ky[ii] * y + kz[ii] * z))
                            x1 =  s[ii]-(x0.T @ rho)
                            # x2 = x1 * cp.conj(x0) / (cp.conj(x0.T) @ x0)
                            x2 = x1 * cp.conj(x0) / (rNx*rNy*rNz)
                            d_rho = lbda * x2
                            rho += d_rho
                            n += 1
                            if n / n_samples > 0.01:
                                m += 1
                                n = 0
                                # print("ART iteration %i: %i %%" % (iteration + 1, m))

                    return rho

        if artMode == 'artPK':
            imgART= artPK(kx_gpu,ky_gpu,kz_gpu,sx_gpu,sy_gpu,sz_gpu,signal_gpu, rho_gpu,lbda, n_iter, index_gpu,timeVec, dBo_gpu)
        elif artMode == 'art':
            imgART= art(kx_gpu,ky_gpu,kz_gpu,sx_gpu,sy_gpu,sz_gpu,signal_gpu, rho_gpu,lbda, n_iter, index_gpu,timeVec, dBo_gpu)
        img = cp.asnumpy(imgART)
        img = np.reshape(img, (1,rNz,rNy,rNx))
        img = np.abs(img).astype(np.float32)
        # print(img.shape)
        return img
    
    def pythonCP():

        signal = kSpace_buffer
        kx = 2*np.pi*kx_buffer
        ky = 2*np.pi*ky_buffer
        kz = 2*np.pi*kz_buffer

        x_vals = np.linspace(-rFOVx / 2 + rFOVx / (2 * rNx) , rFOVx / 2 + rFOVx / (2 * rNx), rNx)
        y_vals = np.linspace(-rFOVy / 2 + rFOVy / (2 * rNy) , rFOVy / 2 + rFOVy / (2 * rNy) , rNy)
        z_vals = np.linspace(-rFOVz / 2 + rFOVz / (2 * rNz) , rFOVz / 2 + rFOVz / (2 * rNz), rNz)
        x, y, z = np.meshgrid(x_vals, y_vals, z_vals,indexing='ij')
        x = x.flatten(order='F') 
        y = y.flatten(order='F')
        z = z.flatten(order='F')

        def boFit(x, y, z):
            return(1.7958916691214196e-06 + 8.780172101062597e-05*(x**1) + 5.1529384399101336e-05*(y**1) + 5.496867688951115e-06*(z**1) 
            + 0.003470212745476351*(x**2) + 0.008922537970838425*(y**1)*(x**1) -0.002658533047887016*(z**1)*(x**1) -0.004877821311734011*(y**2) 
            + 0.011312469067932374*(z**1)*(y**1) -0.004254244444363751*(z**2) + 0.011596056164130399*(x**3) -0.10099062035111861*(y**1)*(x**2) 
            + 0.23066792475669826*(z**1)*(x**2) + 0.06522447284230339*(y**2)*(x**1) -0.09268890836211033*(z**1)*(y**1)*(x**1) -0.2646230382632592*(z**2)*(x**1) 
            + 0.03732888960934208*(y**3) -0.07965855164161945*(z**1)*(y**2) -0.0037222412970749352*(z**2)*(y**1) -0.0990178278290059*(z**3) -0.22386940676842962*(x**4) 
            + 0.4515742630482076*(y**1)*(x**3) -2.427971146354535*(z**1)*(x**3) + 0.8453349553024235*(y**2)*(x**2) + 0.2165221983624357*(z**1)*(y**1)*(x**2) 
            -0.7615407920517288*(z**2)*(x**2) -0.0906496147583209*(y**3)*(x**1) + 1.2421193883567103*(z**1)*(y**2)*(x**1) -1.2839843925917132*(z**2)*(y**1)*(x**1) 
            + 0.8144885330147165*(z**3)*(x**1) + 0.258365771390757*(y**4) -0.5281918562492409*(z**1)*(y**3) + 0.009060285917140476*(z**2)*(y**2) 
            -0.5552447960655633*(z**3)*(y**1) + 0.4740721467059541*(z**4) + 1.5422222012964693*(x**5) + 3.9561465059427485*(y**1)*(x**4) -28.516667829569244*(z**1)*(x**4) 
            -2.8957480824348707*(y**2)*(x**3) -1.3249401492772208*(z**1)*(y**1)*(x**3) + 16.110941965800595*(z**2)*(x**3) + 7.86723677725079*(y**3)*(x**2) 
            -25.429379131759436*(z**1)*(y**2)*(x**2) -20.220814228485104*(z**2)*(y**1)*(x**2) + 2.456315486745019*(z**3)*(x**2) + 1.4570480857538963*(y**4)*(x**1) 
            + 12.843202531324176*(z**1)*(y**3)*(x**1) + 1.5852877967582848*(z**2)*(y**2)*(x**1) + 6.105833933761557*(z**3)*(y**1)*(x**1) + 13.246906094350901*(z**4)*(x**1)
            -3.3671095590823654*(y**5) + 7.685302668620389*(z**1)*(y**4) + 8.679645703874112*(z**2)*(y**3) + 6.056476848223719*(z**3)*(y**2) -4.20548889854099*(z**4)*(y**1)
            + 7.385257017237823*(z**5))

        dBo = boFit(-x, -y, -z)
        dBo = dBo/rdGradAmplitude
        
        rho = np.reshape(np.zeros((rNx*rNy*rNz), dtype=complex), (-1, 1))
        rho = rho[:,0]

        kx_gpu = cp.asarray(kx)
        ky_gpu = cp.asarray(ky)
        kz_gpu = cp.asarray(kz)
        sx_gpu = cp.asarray(x)
        sy_gpu = cp.asarray(y)
        sz_gpu = cp.asarray(z)
        signal_gpu = cp.asarray(signal)
        rho_gpu = cp.asarray(rho)
        dBo_gpu = cp.asarray(dBo)

        def conjugatePhase(kx, ky, kz, x, y, z, s, rho, dBo):
            batch_size = 1000
            # phase = cp.exp(-1j * (cp.outer(kx, x) + cp.outer(ky, y) + cp.outer(kz, z)))
            # rho = cp.dot(s, phase)
            
            for i in range(0, len(x), batch_size):
                # if int(i/len(x)*100) != int((i-1)/len(x)*100):
                    # print(int(i/len(x)*100), ' %')
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                z_batch = z[i:i+batch_size] + dBo[i:i+batch_size]

                phase_batch = cp.exp(1j * (cp.outer(kx, x_batch) + cp.outer(ky, y_batch) + cp.outer(kz, z_batch)))
                rho[i:i+batch_size] = cp.dot(s, phase_batch)
            
            return rho

        imgCP= conjugatePhase(kx_gpu,ky_gpu,kz_gpu,sx_gpu,sy_gpu,sz_gpu,signal_gpu, rho_gpu, dBo_gpu)
        img = cp.asnumpy(imgCP)
        img = np.reshape(img, (1,rNz,rNy,rNx))
        img = np.abs(img).astype(np.float32)
        # print(img.shape)
        return img
    
    if reconMode == 'pythonfft':
        kSpace = np.reshape(kSpace_buffer, [eNx,eNy,eNz])
        kSpace = np.transpose(kSpace, axesOrientation)
        imgRecon = pythonfft(kSpace)
    elif reconMode == 'art':
        imgRecon = pythonART()
    elif reconMode == 'cp':
        imgRecon = pythonCP()
        
    # imgRecon [1,nSl,nPh, nRd]
    # print(imgRecon.shape)
    yield from produce_image(imgRecon)


def reconstruct_mrd_stream(reconMode: str, artMode:str, 
                            input: BinaryIO, output: BinaryIO):
    with mrd.BinaryMrdReader(input) as reader:
        with mrd.BinaryMrdWriter(output) as writer:
            head = reader.read_header()
            if head is None:
                raise Exception("Could not read header")
            writer.write_header(head)
            writer.write_data(
                stream_item_sink(
                    mrdRecon(reconMode, artMode,
                              head, acquisition_reader(reader.read_data()))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstructs an MRD stream")
    parser.add_argument('-i', '--input', type=str, required=False, help="Input file, defaults to stdin")
    parser.add_argument('-o', '--output', type=str, required=False, help="Output file, defaults to stdout")
    parser.add_argument('-r', '--recon', type=str, required=False, help="Reconstruction mode")
    parser.add_argument('-artMode', '--artMode', type=str, default = False, required=False, help="ART mode")
    
    # parser.set_defaults(
    #     input = '/home/tyger/tyger_repo_may/Tyger_MRIlab/input_rdGrad.bin', 
    #     output = '/home/tyger/tyger_repo_may/Tyger_MRIlab/reconART.bin',
    #     recon = 'art', 
    #     artMode = 'art'
    #     )
    args = parser.parse_args()

    input = open(args.input, "rb") if args.input is not None else sys.stdin.buffer
    output = open(args.output, "wb") if args.output is not None else sys.stdout.buffer

    reconstruct_mrd_stream(args.recon, args.artMode,
                            input, output)
