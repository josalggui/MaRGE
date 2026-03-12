#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import copy, os, socket, sys, time, warnings
import multiprocessing as mp

from device import Device
import server_comms as sc
import test_base as tb
from mimo_devices import mimo_dev_run

import local_config as lc

def plot_single(res, rx_t=3.125, title_str=''):
    """ res: list of result tuples in the format [(rxd_iq, msgs), (rxd_iq, msgs), ...]
    rx_t: sampling time used
    title_str: title string to stick at top of plot
    """
    # Assume master is first in the list, slaves are 2nd and thereafter
    rxdm, msgsm = res[0]
    rxds, msgss = res[1]
    rx_len = 0
    for k in [rxdm, rxds]:
        for s in ["rx0", "rx1"]:
            if rx_len == 0:
                rx_len = len(k[s])
            elif rx_len != len(k[s]):
                warnings.warn("RX lengths not all equal -- check timings!")

    plt.figure(figsize=(10, 7))
    # Phases
    plt.subplot(211)
    plt.title(title_str)
    xaxis = np.linspace(0, len(rxdm['rx0']) * rx_t, len(rxdm['rx0']))
    plt.plot(xaxis, np.real(rxdm["rx0"]), label="rx0_real")
    plt.plot(xaxis, np.imag(rxdm["rx0"]), label="rx0_imag")
    plt.ylabel("Master RX0")
    plt.legend()
    plt.grid(True)
    plt.subplot(212)
    xaxis = np.linspace(0, len(rxds['rx1']) * rx_t, len(rxds['rx1']))
    plt.plot(np.real(rxds["rx1"]), label="rx1_real")
    plt.plot(np.imag(rxds["rx1"]), label="rx1_imag")
    plt.ylabel("Slave RX1")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 7))
    # Absolute amplitudes
    plt.subplot(311)
    plt.title(title_str)

    plt.plot(np.abs(rxdm["rx0"]), label="rx0")
    plt.plot(np.abs(rxdm["rx1"]), label="rx1")
    plt.ylabel("Master")
    plt.legend()
    plt.grid(True)
    plt.subplot(312)
    plt.ylabel("Slave")
    plt.plot(np.abs(rxds["rx0"]), label="rx0")
    plt.plot(np.abs(rxds["rx1"]), label="rx1")
    plt.legend()
    plt.grid(True)
    plt.subplot(313)
    plt.ylabel("Master RX0 - slave RX1")

    rxdm0_abs = np.abs(rxdm["rx0"])
    rxds1_abs = np.abs(rxds["rx1"])
    rxdm0_norm = rxdm0_abs / np.mean(rxdm0_abs)
    rxds1_norm = rxds1_abs / np.mean(rxds1_abs)
    plt.plot(rxdm0_abs - rxds1_abs, label="rx0_m - rx1_s")
    plt.plot(rxdm0_norm - rxds1_norm, label="norm(rx0_m) - norm(rx1_s)")
    plt.legend()
    plt.grid(True)

    plt.show()

def plot_repeated(resl, rx_t=3.125, cross_only=True):
    """ resl: list of (rxd_iq, msgs) tuples
    rx_t: RX sampling time used"""
    for res in resl:
        rxdm, rxds = res[0][0], res[1][0]
        if cross_only:
            plt.subplot(211)
        else:
            plt.subplot(222)
            xaxis = np.linspace(0, len(rxdm['rx1']) * rx_t, len(rxdm['rx1']))
            plt.plot(xaxis, np.real(rxdm["rx1"]), 'b', alpha=0.1, label='real')
            plt.plot(xaxis, np.imag(rxdm['rx1']), 'r', alpha=0.1, label='imag')
            plt.xlabel('RX time (us)')
            plt.ylabel("Master RX1")
            plt.legend(['real', 'imag'])
            plt.subplot(221)
        xaxis = np.linspace(0, len(rxdm['rx0']) * rx_t, len(rxdm['rx0']))
        plt.plot(xaxis, np.real(rxdm["rx0"]), 'b', alpha=0.1, label='real')
        plt.plot(xaxis, np.imag(rxdm['rx0']), 'r', alpha=0.1, label='imag')
        plt.xlabel('RX time (us)')
        plt.ylabel("Master RX0")
        plt.legend(['real', 'imag'])
        if cross_only:
            plt.subplot(212)
        else:
            plt.subplot(223)
            xaxis = np.linspace(0, len(rxds['rx0']) * rx_t, len(rxds['rx0']))
            plt.plot(xaxis, np.real(rxds["rx0"]), 'b', alpha=0.1, label='real')
            plt.plot(xaxis, np.imag(rxds["rx0"]), 'r', alpha=0.1, label='imag')
            plt.xlabel('RX time (us)')
            plt.ylabel("Slave RX0")
            plt.legend(['real', 'imag'])
            plt.subplot(224)
        xaxis = np.linspace(0, len(rxds['rx1']) * rx_t, len(rxds['rx1']))
        plt.plot(xaxis, np.real(rxds["rx1"]), 'b', alpha=0.1, label='real')
        plt.plot(xaxis, np.imag(rxds["rx1"]), 'r', alpha=0.1, label='imag')
        plt.xlabel('RX time (us)')
        plt.ylabel("Slave RX1")
        plt.legend(['real', 'imag'])

    file = .amsdflm
    file.save =uafsldjf


def test_mimo_lowlevel(master_ip="localhost", master_port=11111,
                       slave_ip="localhost", slave_port=11112,
                       # existing sockets, if available -- necessary for simulation
                       master_sock=None, slave_sock=None,
                       # when to pulse the output trigger after the start of the master sequence
                       trig_output_time=100e3,
                       # how long the slave takes from being triggered by the
                       # master to beginning its sequence (encapsulates
                       # initialisation, startup pulses etc - should be set by
                       # calibration and rounded to the nearest clock cycle)
                       slave_trig_latency = 6.079,
                       # how long the slave should wait to get triggered (cycles
                       # or 256 x cycles, depending on version), -1 = forever or
                       # until FSM is stopped
                       trig_timeout=10,
                       # how many RX gates to run
                       rx_gates=20,
                       # time to wait between gates
                       rx_gate_interval=1000e3,
                       # how long each RX gate is
                       rx_gate_len=2.0,
                       # what fraction of the RX gate the RF pulse is on for
                       rf_pulse_frac=0.5,
                       # what fraction of the rx_gate_len to move the RF pulse off-centre by
                       rf_pulse_offset=-0.15,
                       # amplitudes and phases of TX0 and TX1 pulses
                       tx0_amp=0.5 + 0.3j,
                       tx1_amp=0.5 + 0.3j,
                       # LO freq, MHz
                       lo_freq = 20,
                       # RX sampling time, us
                       rx_t = 0.0326,
                       plot_preview=False, plot_data=False):
    """Manual 2-board master-slave synchronisation test. Main steps are:
    - start master manually
    - start slave (immediate trigger wait)
    - get master data manually

    Both boards have detection windows on RX0 and RX1, and single RF pulses on
    TX0 and TX1. If we label boards 0/1 as B0 and B1, you should connect them as follows:

    B0_TX0 to B0_RX0
    B0_TX1 to B1_RX1
    B1_TX0 to B1_RX0
    B1_TX1 to B0_RX1

    The returned data should show the same signal on all four channels, down to
    single-cycle timing accuracy. Amplitudes/phases could of course vary,
    depending on the cables you use and fine differences between the channels,
    but the jitter should be below one cycle.
    """

    dev_kwargs = {
        "lo_freq": lo_freq,
        "rx_t": rx_t,
        "print_infos": True,
        "assert_errors": True,
        "halt_and_reset": False,
        "fix_cic_scale": True,
        "set_cic_shift": False,  # needs to be true for open-source cores
        "flush_old_rx": False,
    }

    master_kwargs = {
        'mimo_master': True, 'trig_output_time': trig_output_time, 'slave_trig_latency': slave_trig_latency
        }

    dev_m = Device(
        ip_address=master_ip, port=master_port, prev_socket=master_sock, **(master_kwargs | dev_kwargs)
    )
    dev_s = Device(
        ip_address=slave_ip,
        port=slave_port,
        prev_socket=slave_sock,
        trig_timeout=trig_timeout,
        **dev_kwargs,
    )

    slave_tx_t = np.zeros(2 * rx_gates)
    slave_rx_t = np.zeros_like(slave_tx_t)
    slave_tx0_amp = np.zeros_like(slave_tx_t, dtype=complex)
    slave_tx1_amp = np.zeros_like(slave_tx_t, dtype=complex)
    slave_rx0_en = np.zeros_like(slave_tx_t, dtype=int)
    slave_rx1_en = np.zeros_like(slave_tx_t, dtype=int)

    slave_tx_t_start = (0.5 * (1 - rf_pulse_frac) + rf_pulse_offset) * rx_gate_len
    slave_tx_t_end = (0.5 * (1 + rf_pulse_frac) + rf_pulse_offset) * rx_gate_len
    for gate in range(rx_gates):
        gate_start = gate * (rx_gate_len + rx_gate_interval)
        slave_tx_t[2 * gate] = gate_start + slave_tx_t_start
        slave_tx_t[2 * gate + 1] = gate_start + slave_tx_t_end
        slave_tx0_amp[2 * gate] = tx0_amp
        slave_tx1_amp[2 * gate] = tx1_amp

        slave_rx_t[2 * gate] = gate_start
        slave_rx_t[2 * gate + 1] = gate_start + rx_gate_len
        slave_rx0_en[2 * gate] = 1
        slave_rx1_en[2 * gate] = 1

    # extra ms on the end, to allow for RX to complete
    end_time = rx_gates * (rx_gate_len + rx_gate_interval) + 1e3

    fd = {
        "tx0": (slave_tx_t, slave_tx0_amp),
        "tx1": (slave_tx_t, slave_tx1_amp),
        "tx_gate": (
            np.array([end_time, end_time + 1]),
            np.array([1, 0]),
        ),  # just to have something at the end of the sequence
        "rx0_en": (slave_rx_t, slave_rx0_en),
        "rx1_en": (slave_rx_t, slave_rx1_en),
    }

    dev_m.add_flodict(fd)
    dev_s.add_flodict(fd)

    if plot_preview:
        plt.figure()
        dev_m.plot_sequence()
        plt.figure()
        dev_s.plot_sequence()
        plt.show()

    mpl = [(dev_m, 0), (dev_s, 0)]

    with mp.Pool(2) as p:
        res = p.map(mimo_dev_run, mpl)

    for dev, _ in mpl:
        dev.close_server(only_if_sim=True)
        dev.__del__()  # manual destructor needed

    title_str = f"{rx_gates} RX gates + TX pulses, {rx_gate_interval/1e3:.2f}ms gate interval, {rx_gate_len:.2f}us gate length, {lo_freq:.2f}MHz LO freq, {1/rx_t:.2f}MHz sample rate"
    if plot_data:
        plot_single(res, rx_t, title_str)

    return res


def test_single_sim(plot_data=False):
    tb.base_setup_class()
    pm, sm = tb.base_setup(  # master
        fst_dump=False, csv_path=os.path.join("/tmp", "marga_sim_m.csv"), port=11111
    )
    ps, ss = tb.base_setup(  # slave
        fst_dump=False, csv_path=os.path.join("/tmp", "marga_sim_s.csv"), port=11112
    )

    test_mimo_lowlevel(master_ip="localhost", master_port=11111,
                       slave_ip="localhost", slave_port=11112,
                       master_sock=sm, slave_sock=ss,
                       trig_output_time=1,
                       rx_gates=100,
                       rx_gate_interval=0.1,
                       plot_preview=False, plot_data=plot_data)

    # halt simulation
    sc.send_packet(sc.construct_packet({}, 0, command=sc.close_server_pkt), sm)
    sc.send_packet(sc.construct_packet({}, 0, command=sc.close_server_pkt), ss)
    sm.close()
    ss.close()
    pm.wait(1)  # wait a short time for simulator to close
    ps.wait(1)


def test_single(rx_gates=10, rx_gate_interval=1000e3, plot_data=False,
                trig_timeout=136533, **kwargs):
    return test_mimo_lowlevel(rx_gates=rx_gates,
                              rx_gate_interval=rx_gate_interval,
                              trig_timeout=trig_timeout,
                              plot_data=plot_data, **kwargs)


def test_repeated(reps=10, plot_persistent=False,
                  rx_t=0.0326,
                  master_ip="192.168.1.160",
                  master_port=11111,
                  slave_ip="192.168.1.158",
                  slave_port=11111,
                  **kwargs):

    resl = []
    ms = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ms.connect((master_ip, master_port))
    ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ss.connect((slave_ip, slave_port))

    for rep in range(reps):
        print(f"Sequence repetition {rep}")
        res = test_single(master_sock=ms, slave_sock=ss, rx_t=rx_t, **kwargs)
        resl.append(res)

    ms.close()
    ss.close()

    if plot_persistent:
        plot_repeated(resl, rx_t)


if __name__ == "__main__":
    test_single_simulation = False
    test_single_real = True
    test_repeated_real = False

    ## Check that libraries etc are all correctly configured (just simulation)
    if test_single_simulation:
        test_single_sim(plot_data=True)

    ## Check basic operation and sync
    if test_single_real:
            test_single(rx_gates=10, rx_gate_interval=50e3, rx_gate_len=2,
                        slave_trig_latency=6.079,
                        plot_data=True, plot_preview=False,
                        master_ip=lc.ip_address, master_port=lc.port,
                        slave_ip=lc.ip_address_slave, slave_port=lc.port_slave)

    ## Check repeatability over multiple rounds
    if test_repeated_real:
        rx_gates = 5
        reps = 40
        params_shared = {'reps': reps, 'rx_gates': rx_gates, 'rx_gate_interval': 1e3,
                         'plot_persistent': True,
                         'master_ip': lc.ip_address, 'master_port': lc.port,
                         'slave_ip': lc.ip_address_slave, 'slave_port':lc.port_slave,
                         'rx_gate_len': 10e3, 'rx_t': 30,
                         'rf_pulse_offset': 0}

        params_unsynced = {'trig_timeout': 0, 'trig_output_time': 1}
        params_synced = {'trig_timeout': 100000, 'trig_output_time': 10e3}

        plt.figure(figsize=(10,7))
        test_repeated(**(params_unsynced | params_shared))

        plt.figure(figsize=(10,7))
        test_repeated(**(params_synced | params_shared))

        plt.show()
