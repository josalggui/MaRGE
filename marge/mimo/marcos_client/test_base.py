#!/usr/bin/env python3
#
# Base definitions and functions used for marga testing/simulation

import sys, os, subprocess, logging, warnings, socket, unittest, time, inspect
import numpy as np
import matplotlib.pyplot as plt

import server_comms as sc

import marcompile as mc
import device as dev

# Logger: uncomment to save test logging information to file
if False:
    logging.basicConfig(filename="test_base.log", level=logging.DEBUG)

# PDB shorthand
import pdb

st = pdb.set_trace


# this must happen after the mc and exp imports
fpga_clk_freq_MHz = 122.88  # only simulate tests for SDRLab-122
mc.fpga_clk_freq_MHz = fpga_clk_freq_MHz
dev.fpga_clk_freq_MHz = fpga_clk_freq_MHz

# simulation configuration
marga_sim_path = os.path.join("..", "marga")
marga_sim_csv = os.path.join("/tmp", "marga_sim.csv")

# Set to True to debug with GTKWave -- just make sure you only run one test at a time!
marga_sim_fst_dump = False
marga_sim_fst = os.path.join("/tmp", "marga_sim.fst")

# Arguments for compare_csv when running gradient tests
fhd_config = {
    'initial_bufs': np.array([
        # see marga.sv, gradient control lines (lines 186-190, 05.02.2021)
        # strobe for both LSB and LSB, reset_n = 1, spi div = 10, grad board select (1 = ocra1, 2 = gpa-fhdo)
        (1 << 9) | (1 << 8) | (10 << 2) | 2,
        0, 0,
        0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0], dtype=np.uint16),
    'latencies': np.array([
        0, 276, 276, # grad latencies match SPI div
        0, 0, # rx
        0, 0, 0, 0, # tx
        0, 0, 0, 0, 0, 0, # lo phase
        0, 0 # gates and LEDs, RX config
    ], dtype=np.uint16)}

oc1_config = {
    'initial_bufs': np.array([
        # see marga.sv, gradient control lines (lines 186-190, 05.02.2021)
        # strobe for both LSB and LSB, reset_n = 1, spi div = 10, grad board select (1 = ocra1, 2 = gpa-fhdo)
        (1 << 9) | (1 << 8) | (10 << 2) | 1,
        0, 0,
        0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0], dtype=np.uint16),
    'latencies': np.array([
        0, 268, 268, # grad latencies match SPI div
        0, 0, # rx
        0, 0, 0, 0, # tx
        0, 0, 0, 0, 0, 0, # lo phase
        0, 0 # gates and LEDs, RX config
    ], dtype=np.uint16)}

gb_orig = None
gb_changed = False

def base_setup_class():
    # TODO make this check for a file first
    subprocess.call(["make", "-j4", "-s", "-C", os.path.join(marga_sim_path, "build")])
    subprocess.call(["fallocate", "-l", "516KiB", "/tmp/marcos_server_mem"])
    # in case other instances were started earlier
    subprocess.call(["killall", "marga_sim"], stderr=subprocess.DEVNULL)

    warnings.simplefilter("ignore", mc.MarServerWarning)


def base_setup(fst_dump=marga_sim_fst_dump, csv_path=marga_sim_csv, fst_path=marga_sim_fst, port=11111):
    # start simulation
    if marga_sim_fst_dump:
        p = subprocess.Popen([os.path.join(marga_sim_path, "build", "marga_sim"), "csv=" + csv_path, "fst=" + fst_path, "port=" + str(port)],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.STDOUT)
    else:
        p = subprocess.Popen([os.path.join(marga_sim_path, "build", "marga_sim"), "csv=" + marga_sim_csv, "port=" + str(port)],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.STDOUT)


    # open socket
    time.sleep(0.05) # give marga_sim time to start up

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("localhost", port)) # only connect to local simulator
    return p, s


def base_teardown(process, socket):
    # process.terminate() # if not already terminated
    # process.kill() # if not already terminated
    socket.close()

    if marga_sim_fst_dump:
        # open GTKWave
        os.system(
            "gtkwave "
            + marga_sim_fst
            + " "
            + os.path.join(marga_sim_path, "src", "marga_sim.sav")
        )


def set_grad_board(gb):
    global gb_orig, gb_changed
    if not gb_changed:
        gb_orig = mc.grad_board
    dev.lc.grad_board = gb
    mc.grad_board = gb
    gb_changed = True


def restore_grad_board():
    global gb_orig, gb_changed
    dev.lc.grad_board = gb_orig
    mc.grad_board = gb_orig
    gb_changed = False


def sanitise_arrays(rdata, sdata):
    rdata[1:, 0] -= rdata[1, 0]  # subtract off initial offset time
    sdata[1:, 0] -= sdata[1, 0]  # subtract off initial offset time

    csv_v03_cols = 31  # columns in v0.3 of CSV format
    csv_v02_cols = 25  # columns in v0.2 of CSV format
    rdata_cols = rdata.shape[1]
    sdata_cols = sdata.shape[1]
    rdata_v03 = rdata_cols == csv_v03_cols
    sdata_v03 = sdata_cols == csv_v03_cols

    # both sets of CSV data must be converted to same format
    if rdata_v03 != sdata_v03 and (
        (rdata_cols == csv_v02_cols) or (sdata_cols == csv_v02_cols)
    ):
        logging.info(
            f"{inspect.getouterframes(inspect.currentframe(), 1)[2][3]}: Incompatible CSV formats. Attempting to compare only v0.2 subset."
        )

        def v03tov02(mat, mat_old):
            # convert matrix format from CSV v0.3 (DDS included) to CSV v0.2 (DDS not included)
            if (
                np.array_equal(mat[:2, 0], mat_old[:2, 0])
                and np.array_equal(mat[3:, 0], mat_old[2:, 0])
                and mat.shape[0] > 2
                and mat[2, 0] == 1
            ):
                mat = np.vstack(
                    [mat[:2, :], mat[3:, :]]
                )  # remove row caused by DDS phase clear, but check that the rest of the diffs match precisely
            return mat[:, :25]

        if rdata_v03:
            rdata = v03tov02(rdata, sdata)

        elif sdata_v03:
            sdata = v03tov02(sdata, rdata)

    return rdata, sdata


def compare_csv(
    fname,
    sock,
    proc,
    initial_bufs=np.zeros(mc.MARGA_BUFS, dtype=np.uint16),
    latencies=np.zeros(mc.MARGA_BUFS, dtype=np.uint32),
    self_ref=True,  # use the CSV source file as the reference file to compare the output with
):

    source_csv = os.path.join("csvs", fname + ".csv")
    lc = mc.csv2bin(
        source_csv, quick_start=False, initial_bufs=initial_bufs, latencies=latencies
    )
    data = np.array(lc, dtype=np.uint32)

    # run simulation
    rx_data, msgs = sc.command({"run_seq": data.tobytes()}, sock)

    # halt simulation
    sc.send_packet(sc.construct_packet({}, 0, command=sc.close_server_pkt), sock)
    sock.close()
    proc.wait(1)  # wait a short time for simulator to close

    # compare resultant CSV with the reference
    if self_ref:
        rdata = np.loadtxt(source_csv, skiprows=1, delimiter=",", comments="#").astype(
            np.uint32
        )
        sdata = np.loadtxt(
            marga_sim_csv, skiprows=1, delimiter=",", comments="#"
        ).astype(np.uint32)
        rdata, sdata = sanitise_arrays(rdata, sdata)
        return rdata.tolist(), sdata.tolist()
    else:
        ref_csv = os.path.join("csvs", "ref_" + fname + ".csv")
        with open(ref_csv, "r") as ref:
            refl = ref.read().splitlines()
        with open(marga_sim_csv, "r") as sim:
            siml = sim.read().splitlines()
        return refl, siml


def compare_dict(
    source_dict,
    ref_fname,
    sock,
    proc,
    initial_bufs=np.zeros(mc.MARGA_BUFS, dtype=np.uint16),
    latencies=np.zeros(mc.MARGA_BUFS, dtype=np.uint32),
    ignore_start_delay=True,
):

    lc = mc.dict2bin(source_dict, initial_bufs=initial_bufs, latencies=latencies)
    data = np.array(lc, dtype=np.uint32)

    # run simulation
    rx_data, msgs = sc.command({"run_seq": data.tobytes()}, sock)

    # halt simulation
    sc.send_packet(sc.construct_packet({}, 0, command=sc.close_server_pkt), sock)
    sock.close()
    proc.wait(1)  # wait a short time for simulator to close

    ref_csv = os.path.join("csvs", ref_fname + ".csv")
    with open(ref_csv, "r") as ref:
        refl = ref.read().splitlines()
    with open(marga_sim_csv, "r") as sim:
        siml = sim.read().splitlines()
    # return refl, siml

    ref_csv = os.path.join("csvs", ref_fname + ".csv")
    if ignore_start_delay:
        rdata = np.loadtxt(ref_csv, skiprows=1, delimiter=",", comments="#").astype(
            np.uint32
        )
        sdata = np.loadtxt(
            marga_sim_csv, skiprows=1, delimiter=",", comments="#"
        ).astype(np.uint32)
        rdata, sdata = sanitise_arrays(rdata, sdata)
        return rdata.tolist(), sdata.tolist()
    else:
        with open(ref_csv, "r") as ref:
            refl = ref.read().splitlines()
        with open(marga_sim_csv, "r") as sim:
            siml = sim.read().splitlines()
        return refl, siml


def dev_run(d):
    """Function for customising how compare_dev_dict() runs Device tests;
    e.g. for testing different Device methods etc (see test_lo_change_dev in
    test_marga_model.py for an example)

    """
    rx_data, msgs = d.run()
    return rx_data, msgs


def compare_dev_dict(
    source_dict,
    ref_fname,
    sock,
    proc,
    # initial_bufs=np.zeros(mc.MARGA_BUFS, dtype=np.uint16),
    # latencies=np.zeros(mc.MARGA_BUFS, dtype=np.uint32),
    ignore_start_delay=True,
    run_fn=dev_run,
    port=11111,
    **kwargs,
):
    """Arguments the same as for compare_dict(), except that the source
    dictionary is in floating-point units, and the kwargs are passed
    to the Device class constructor. Note that the initial_bufs
    and latencies are supplied to the Device class from the
    classes in grad_board.py.
    """

    lo_freq = (
        1234567890 * 122.88 / 2**31
    )  # Arbitrary default LO frequency for unit tests
    d = dev.Device(
        ip_address="localhost",
        port=port,
        lo_freq=lo_freq,
        prev_socket=sock,
        seq_dict=source_dict,
        **kwargs,
    )

    # run simulation
    rx_data, msgs = run_fn(d)

    # halt simulation
    sc.send_packet(sc.construct_packet({}, 0, command=sc.close_server_pkt), sock)
    sock.close()
    proc.wait(1)  # wait a short time for simulator to close

    ref_csv = os.path.join("csvs", ref_fname + ".csv")
    with open(ref_csv, "r") as ref:
        refl = ref.read().splitlines()
    with open(marga_sim_csv, "r") as sim:
        siml = sim.read().splitlines()
    # return refl, siml

    ref_csv = os.path.join("csvs", ref_fname + ".csv")
    if ignore_start_delay:
        rdata = np.loadtxt(ref_csv, skiprows=1, delimiter=",", comments="#").astype(
            np.uint32
        )
        sdata = np.loadtxt(
            marga_sim_csv, skiprows=1, delimiter=",", comments="#"
        ).astype(np.uint32)
        rdata, sdata = sanitise_arrays(rdata, sdata)
        return rdata.tolist(), sdata.tolist()
    else:
        with open(ref_csv, "r") as ref:
            refl = ref.read().splitlines()
        with open(marga_sim_csv, "r") as sim:
            siml = sim.read().splitlines()
        return refl, siml
