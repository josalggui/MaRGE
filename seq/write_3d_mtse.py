import math
import warnings

import numpy as np

from matplotlib import pyplot as plt

import pypulseq as pp
#import sys
#sys.path.append('/home/zaitsev/pulseq_home/pypulseq')
#import pypulseq as pp
#del sys.path[-1]


def main(plot: bool, write_seq: bool, seq_filename: str = "mtse_3d_pypulseq.seq"):
    # ======
    # SETUP
    # ======
    
    # Set system limits
    system = pp.Opts(
        max_grad=10,
        grad_unit="mT/m",
        max_slew=100,
        slew_unit="T/m/s",
        rf_ringdown_time=15e-6,
        rf_dead_time=15e-6,
        adc_dead_time=0e-6,
    )

    seq = pp.Sequence(system)  # Create a new sequence object

    # basic parameters
    fov_mm     = (256, 192, 128) # Define FOV in mm
    Nx, Ny, Nz = 40, 30, 20      # Define resolution (matrix sizes)
    n_echo = 10  # Number of echoes
    TE = 10e-3 # echo time of the first spin echo
    TR = 2000e-3  # Repetition time

    dG = 500e-6
    sampling_time = 4e-3
    ro_flattop_add=500e-6
    Nx_pre=10
    Nx_post=Nx_pre
    os=5
    t_ex  = 60e-6 # needs to be of the same "parity" as the t_ref (fixme?)
    t_ref = 100e-6
    fsp_r = 1
    fsp_s = 0.5

    # derived and modieifed parameters
    fov = np.array(fov_mm)*1e-3
    TE = round(TE/system.grad_raster_time/2)*system.grad_raster_time*2 # TE (=ESP) should be divisible to a double gradient raster, which simplifies calcuations
    ro_flattop_time=sampling_time+2*ro_flattop_add;
    rf_add = math.ceil(max(system.rf_dead_time,system.rf_ringdown_time)/system.grad_raster_time)*system.grad_raster_time # round up dead times to the gradient raster time to enable correct TE & ESP calculation
    t_sp = round((0.5 * (TE - ro_flattop_time - t_ref) - rf_add)/system.grad_raster_time)*system.grad_raster_time
    t_spex = round((0.5 * (TE - t_ex - t_ref) - 2*rf_add)/system.grad_raster_time)*system.grad_raster_time
    #print("rf_add", rf_add)
    
    rf_ex_phase = np.pi / 2
    rf_ref_phase = 0

    # ======
    # CREATE EVENTS
    # ======
    flip_ex = 90 * np.pi / 180
    rf_ex = pp.make_block_pulse(
        flip_angle=flip_ex,
        system=system,
        duration=t_ex,
        delay=rf_add, 
        phase_offset=rf_ex_phase,
    )
    d_ex=pp.make_delay(t_ex+rf_add*2)
    
    flip_ref = 180 * np.pi / 180
    rf_ref = pp.make_block_pulse(
        flip_angle=flip_ref,
        system=system,
        duration=t_ref,
        delay=rf_add, 
        phase_offset=rf_ref_phase,
        use="refocusing",
    )
    d_ref=pp.make_delay(t_ref+rf_add*2)
    
    delta_kx = 1 / fov[0]
    ro_amp = Nx * delta_kx / sampling_time

    gr_acq = pp.make_trapezoid(
        channel="x",
        system=system,
        amplitude = ro_amp,
        flat_time=ro_flattop_time,
        delay=t_sp,
        rise_time=dG,
    )
    adc = pp.make_adc(
        num_samples=(Nx_pre+Nx+Nx_post)*os, dwell=sampling_time/Nx/os, delay=t_sp+dG-Nx_pre*sampling_time/Nx
    )
    gr_spr = pp.make_trapezoid(
        channel="x",
        system=system,
        area=gr_acq.area * fsp_r,
        duration=t_sp,
        rise_time=dG,
    )

    agr_spr = gr_spr.area
    agr_preph = gr_acq.area / 2 + agr_spr
    gr_preph = pp.make_trapezoid(
        channel="x", system=system, area=agr_preph, duration=t_spex, rise_time=dG
    )
    # Phase-encoding
    delta_ky = 1 / fov[1]
    gp_max = pp.make_trapezoid(
                    channel="y",
                    system=system,
                    area=delta_ky*Ny/2,
                    duration=t_sp,
                    rise_time=dG,
                )
    delta_kz = 1 / fov[2]
    gs_max = pp.make_trapezoid(
                    channel="z",
                    system=system,
                    area=delta_kz*Nz/2,
                    duration=t_sp,
                    rise_time=dG,
                )

    # combine parts of the read gradient
    gc_times = np.array(
        [
            0,
            gr_spr.rise_time,
            gr_spr.flat_time,
            gr_spr.fall_time,
            gr_acq.flat_time,
            gr_spr.fall_time,
            gr_spr.flat_time,
            gr_spr.rise_time,
        ])
    gc_times = np.cumsum(gc_times)

    gr_amp = np.array([0, gr_spr.amplitude, gr_spr.amplitude, gr_acq.amplitude, gr_acq.amplitude, gr_spr.amplitude, gr_spr.amplitude, 0])
    gr = pp.make_extended_trapezoid(channel="x", times=gc_times, amplitudes=gr_amp)

    gp_amp = np.array([0, gp_max.amplitude, gp_max.amplitude, 0, 0, -gp_max.amplitude, -gp_max.amplitude, 0])
    gp_max = pp.make_extended_trapezoid(channel="y", times=gc_times, amplitudes=gp_amp)

    gs_amp = np.array([0, gs_max.amplitude, gs_max.amplitude, 0, 0, -gs_max.amplitude, -gs_max.amplitude, 0])
    gs_max = pp.make_extended_trapezoid(channel="z", times=gc_times, amplitudes=gs_amp)

    #print("t_spex", t_spex*1e6, "gr_preph", pp.calc_duration(gr_preph)*1e6)
    #print("t_sp", t_sp*1e6, "gc_times", gc_times*1e6)
    
    # Fill-times
    t_ex = pp.calc_duration(d_ex) + pp.calc_duration(gr_preph)
    t_ref = pp.calc_duration(d_ref) + pp.calc_duration(gr)

    t_train = t_ex + n_echo * t_ref

    TR_fill = TR - t_train
    # Round to gradient raster
    TR_fill = system.grad_raster_time * np.round(TR_fill / system.grad_raster_time)
    if TR_fill < 0:
        TR_fill = 1e-3
        warnings.warn(
            f"TR too short, adapted to: {1000 * (TE_train + TR_fill)} ms"
        )
    else:
        print(f"TR fill: {1000 * TR_fill} ms")
    delay_TR = pp.make_delay(TR_fill)

    # ======
    # CONSTRUCT SEQUENCE
    # ======
    for Cz in range(-1,Nz):
        if Cz >= 0:
            sl_scale = (Cz-Nz/2)/Nz*2;
            Ny_range=range(Ny)
        else:
            sl_scale = 0.0
            Ny_range=range(1) # skip the Ny loop for dummy scan(s)
            
        gs=pp.scale_grad(gs_max, sl_scale)
            
        for Cy in Ny_range:
            seq.add_block(rf_ex, d_ex)
            #print("1", seq.block_durations[1]*1e6)
            seq.add_block(gr_preph)
            #print("2", seq.block_durations[2]*1e6)

            if Cz >= 0:
                pe_scale = (Cy-Ny/2)/Ny*2;
            else:
                pe_scale = 0.0
                
            gp=pp.scale_grad(gp_max, pe_scale)

            for k_echo in range(n_echo):
                
                seq.add_block(rf_ref, d_ref)
                #print("3", seq.block_durations[3]*1e6)
                if Cz >= 0:
                    seq.add_block(gs, gp, gr, adc)
                else:
                    seq.add_block(gs, gp, gr)
                #print("4", seq.block_durations[4]*1e6)
                
            seq.add_block(delay_TR)

    (
        ok,
        error_report,
    ) = seq.check_timing()  # Check whether the timing of the sequence is correct
    if ok:
        print("Timing check passed successfully")
    else:
        print("Timing check failed. Error listing follows:")
        [print(e) for e in error_report]

    # ======
    # VISUALIZATION
    # ======
    if plot:
        seq.plot(time_range=(0, TR*3))

        [k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc] = seq.calculate_kspace() #(gradient_offset=(3000,-2000,2000))

        t_adc_echo=t_adc[round((Nx_pre+Nx+Nx_post)*os/2)]
        n_ex_echo=np.where(t_excitation<t_adc_echo)[0][-1]
        n_ref_echo=np.where(t_refocusing<t_adc_echo)[0][-1]
        print("desired TE=",TE,
              " realized TE=", t_adc_echo-t_excitation[n_ex_echo],
              " error of TE=", t_adc_echo-t_excitation[n_ex_echo]-TE,
              " t_ref-t_ex=", t_refocusing[n_ref_echo]-t_excitation[n_ex_echo])
        print("Echo Spacing=", t_refocusing[1]-t_refocusing[0],
              " error of ESP=", t_refocusing[1]-t_refocusing[0]-TE)
        print("desired TR=", TR,
              " achieved TR=", t_excitation[1]-t_excitation[0],
              " error of TR=", t_excitation[1]-t_excitation[0]-TR) 

        n1=100000
        n2=10000
        plt.figure()
        plt.plot(k_traj[0,0:n1],k_traj[1,0:n1], 'b-')
        plt.plot(k_traj_adc[0,0:n2],k_traj_adc[1,0:n2], 'r.')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.title('k-space trajectory')
        plt.xlabel(r'$k_x \mathregular{\ /m^{-1}}$')
        plt.ylabel(r'$k_y \mathregular{\ /m^{-1}}$')
        plt.show()

        plt.figure()
        plt.plot(k_traj[0,0:n1])
        plt.plot(k_traj[1,0:n1])
        plt.plot(k_traj[2,0:n1])
        ax = plt.gca()
        ax.grid();
        plt.title('k-space trajectory')
        plt.xlabel(r'$k_x \mathregular{\ /m^{-1}}$')
        plt.ylabel(r'$k_y \mathregular{\ /m^{-1}}$')
        plt.show()

    # =========
    # WRITE .SEQ
    # =========
    if write_seq:
        seq.write(seq_filename)


if __name__ == "__main__":
    main(plot=True, write_seq=True)
