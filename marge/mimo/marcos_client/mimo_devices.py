#!/usr/bin/env python3
#
# Utility class to operate MIMO systems

import socket, time, warnings
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

import local_config as lc
import grad_board as gb
import server_comms as sc
import marcompile as mc

from device import Device


def mimo_dev_run(devt):
    """Allows the parallelisation of Device run() calls, which is essential
    since the slaves will block otherwise."""
    dev, delay = devt
    time.sleep(delay)
    rxd, msgs = dev.run()
    return rxd, msgs


class MimoDevices:
    """Manages multiple Devices, collates/distributes settings between them and
    handles messages and warnings."""

    def __init__(self, ips, ports, trig_output_time=10e3, slave_trig_latency=6.079,
                 trig_timeout=136533, master_run_delay=0, extra_args=None, **kwargs):
        """ips: list of device IPs, master first

        ports: list of device ports, master first

        trig_output_time: usec, when to trigger the slaves after beginning the
        sequence on the master

        trig_latency: usec, how long the slaves take from being triggered by the
        master to beginning their sequences (plus any additional I/O or cable
        latencies)

        trig_timeout: usec, how long should the slaves wait for a trigger until
        they run their preprogrammed sequences anyway. Same behaviour and
        limitations as for the Device class. Negative values = infinite timeout
        so only use this when the system is debugged.

        master_run_delay: sec, how long to wait for the slaves to start running
        before starting the master compilation/programming/execution -- the
        master must begin after the slaves are awaiting a trigger, otherwise
        sync will not be maintained. Positive values will delay the master's
        run() call, negative values will delay the slaves. [TODO: Also accepts a
        per-device list.]

        extra_args: list of dictionaries of extra arguments to each Device
        object, master first

        All remaining arguments supported by the Device class are also
        supported, and will be passed down to each Device.

        """

        devN = len(ips)
        assert len(ips) == len(ports), f" Supplied {len(ips)} IPs but {len(ports)} ports"

        if extra_args is None:
            device_args = [kwargs] * devN
        else:
            assert len(ips) == len(extra_args), f" Supplied {len(ips)} IPs but {len(ports)} extra arg dicts"
            device_args = list(ea | kwargs for ea in extra_args)

        master_rd = 0
        slave_rd = 0
        if master_run_delay > 0:
            master_rd = master_run_delay
        else:
            slave_rd = -master_run_delay

        self._devs = []
        self._pool_args = []

        for k, (ip, port, devargs) in enumerate(zip(ips, ports, device_args)):
            if k == 0:
                # TODO cannot yet handle the case where the MIMO system is externally triggered
                kwargs = devargs | {
                    'mimo_master': True,
                    'trig_timeout': 0,
                    'trig_output_time': trig_output_time }
                run_delay = master_rd
            else:
                kwargs = devargs | {
                    'trig_timeout': trig_timeout
                }
                run_delay = slave_rd

            dev = Device(
                ip_address=ip, port=port, **kwargs)

            self._devs.append(dev)
            self._pool_args.append((dev, run_delay))

    def get_device(self, k):
        return self._devs[k]

    # shorthand
    def dev(self, k):
        return self.get_device(self, k)

    def dev_list(self):
        return self._devs

    def run(self):
        """ Runs the Devices in parallel, collates their results and settings """
        with mp.Pool(len(self._devs)) as p:
            res = p.map(mimo_dev_run, self._pool_args)

        return res


def test_mimo_devices(single=True, reps=1, **kwargs):
    # importing here to avoid a circular import - these are just throwaway plotting functions
    from test_mimo_lowlevel import plot_single, plot_repeated

    ips = [lc.ip_address, lc.ip_address_slave]
    ports = [lc.port, lc.port_slave]

    # Manually handle sockets if repeated tests are being run
    socks = []
    for ip, port in zip(ips, ports):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((ip, port))
        socks.append(sock)

    # How to do per-Device extra arguments

    # either
    extra_args = list({'prev_socket': sock } for sock in socks)

    # or
    extra_args = [{'prev_socket': socks[0]}, {'prev_socket': socks[1]}]

    # How to do global arguments (shared for every device)
    rx_t = 1

    mdev = MimoDevices(ips=ips, ports=ports, extra_args=extra_args, rx_t=rx_t, **kwargs)
    devs = mdev.dev_list()

    for dev in devs:
        dev.add_flodict({
            'tx0': (np.array([10, 30, 80, 90]), np.array([0.3+0.5j, 0, 0.8+0.3j, 0])),
            'tx1': (np.array([20, 40, 75, 85]), np.array([0.3+0.5j, 0, 0.4+0.2j, 0])),
            'rx0_en': (np.array([0, 100]), np.array([1, 0])),
            'rx1_en': (np.array([0, 100]), np.array([1, 0])),
            })

    if single:
        res = mdev.run()
        plot_single(res, rx_t)
        assert reps == 1, "reps must be 1 for single test"
    else:
        resl = []
        for k in range(reps):
            print(f"Round {k+1}")
            res = mdev.run()
            resl.append(res)

        plot_repeated(resl, rx_t, cross_only=True)

    plt.show()
    # Only necessary if the socket isn't being closed anyway
    (dev.__del__() for dev in devs)


if __name__ == "__main__":
    ## Single test
    # test_mimo_devices(single=True)

    ## 100x repetitions, default trig output time, default trig timeout
    test_mimo_devices(single=False, reps=100)

    ## 5x repetitions, 1-second trig output time, infinite trig timeout
    # test_mimo_devices(single=False, reps=5, trig_output_time=1e6, trig_timeout=-1)
