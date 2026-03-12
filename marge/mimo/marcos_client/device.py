#!/usr/bin/env python3
#
# Basic toolbox for server operations; wraps up a lot of stuff to avoid the need for hardcoding on the user's side.

import socket, time, warnings
import numpy as np
import matplotlib.pyplot as plt

import local_config as lc
import grad_board as gb
import server_comms as sc
import marcompile as mc


class Device:
    """Wrapper class for managing a hardware interface to an SDRLab
    ip_address: IP address of SDRLab

    port: port of SDRLab

    fpga_clk_freq_MHz: SDRLab's clock frequency

    grad_board: gradient board of SDRLab; either "ocra1", "gpa-fhdo" or "none". ## TODO enable NONE functionality for grad_board

    lo_freq: local oscillator frequencies, MHz: either single float,
    iterable of two or iterable of three values. Control three
    independent LOs. At least a single float must be supplied.
    If supplying 2 or 3 values, must also specify the rx_lo.

    rx_t: RF RX sampling time/s in microseconds; single float or tuple
    of two.

    rx_lo: RX local oscillator sources (integers): 0 - 2 correspond to
    the three LO NCOs, 3 is DC. Single integer or tuple of two.  By
    default will use local oscillator 0 for both RX channels, unless
    otherwise specified.

    Note: TX0 and TX1 will always be assigned to NCOs 0 and 1. NCO 2
    is for free adjustment of the RX frequency relative to the TX.

    csv_file: path to a CSV execution file, which will be
    compiled with marcompile.py . If this is not supplied, the
    sequence bytecode should be supplied manually using
    define_sequence() before run() is called.

    OPTIONAL PARAMETERS - ONLY ALTER IF YOU KNOW WHAT YOU'RE DOING

    grad_max_update_rate: used to calculate the frequency to run the
    gradient SPI interface at - must be high enough to support your
    maximum gradient sample rate but not so high that you experience
    communication issues. Leave this alone unless you know what you're
    doing.

    print_infos: print debugging messages from server to stdout

    assert_errors: errors returned from the server will be treated as
    exceptions by the class, halting the program

    init_gpa: initialise the GPA during the construction of this class
    """

    def __init__(
        self,
        ip_address=lc.ip_address,  # by default imported from local_config, but can be overriden if desired
        port=lc.port,  # by default imported from local_config, but can be overriden if desired
        fpga_clk_freq_MHz=lc.fpga_clk_freq_MHz,
        grad_board=lc.grad_board,
        lo_freq=1,  # MHz
        rx_t=3.125,  # us; multiples of 1/122.88, such as 3.125, are exact, others will be rounded to the nearest multiple of the 122.88 MHz clock
        seq_dict=None,  # provide upon construction rather than using add_flodict() later
        seq_csv=None,  # provide upon construction rather than using add_flodict() later
        rx_lo=0,  # which of internal NCO local oscillators (LOs), out of 0, 1, 2, to use for each channel
        grad_max_update_rate=0.2,  # MSPS, across all channels in parallel, best-effort
        grad_latency=0,  # us, delay inherent in gradient board + external hardware (positive values are corrected for by shifting the gradient update events back in time)
        gpa_fhdo_offset_time=0,  # when GPA-FHDO is used, offset the Y, Z and Z2 gradient times by 1x, 2x and 3x this value to emulate 'simultaneous' updates
        print_infos=True,  # show server info messages
        assert_errors=True,  # halt on server errors
        init_gpa=False,  # initialise the GPA (will reset its outputs when the Device object is created)
        initial_wait=None,  # initial pause before experiment begins - required to configure the LOs and RX rate; must be at least a few us. Is suitably set based on grad_max_update_rate by default.
        auto_leds=True,  # automatically scan the LED pattern from 0 to 255 as the sequence runs (set to off if you wish to manually control the LEDs)
        mimo_master=False,  # if MIMO master, add a trigger pulse to trig_out channel
        trig_output_time=0, # usec, when should the master's trig_out pulse occur
        slave_trig_latency=6.079,  # usec, how much to delay MIMO master events after the trigger is sent to slaves. Only used for MIMO masters, MIMO slaves or standalone devices unaffected. Device, trigger method and cable-specific.
        trig_timeout=0,  # if MIMO slave or externally-triggered master, nonzero values will add a wait-for-trigger instruction to the start of the sequence, with a timeout in microseconds. Positive values must be below 136533 (0.13sec), which is the current limit of marga. Negative values will lead to an infinite wait (i.e. never timing out, blocking forever until a trigger arrives.)
        prev_socket=None,  # previously-opened socket, if want to maintain status, running a simulation, etc
        fix_cic_scale=True,  # scale the RX data precisely based on the rate being used; otherwise a 2x variation possible in data amplitude based on rate
        set_cic_shift=False,  # program the CIC internal bit shift to maintain the gain within a factor of 2 independent of rate; required if the open-source CIC is used in the design
        allow_user_init_cfg=False,  # allow user-defined alteration of marga configuration set by init, namely RX rate, LO properties etc; see the compile() method for details
        halt_and_reset=False,  # upon connecting to the server, halt any existing sequences that may be running
        flush_old_rx=False,  # when debugging or developing new code, you may accidentally fill up the RX FIFOs - they will not automatically be cleared in case there is important data inside. Setting this true will always read them out and clear them before running a sequence. More advanced manual code can read RX from existing sequences.
    ):
        self._ip_address = ip_address
        self._port = port
        self._fpga_clk_freq_MHz = fpga_clk_freq_MHz
        self._grad_board = grad_board

        # create socket early so that destructor works
        self._close_socket = True
        if prev_socket is None:
            self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._s.connect((ip_address, port))
        else:
            self._s = prev_socket
            self._close_socket = False  # do not close previous socket

        self.set_lo_freq(lo_freq)

        if not hasattr(rx_t, "__len__"):
            rx_t = rx_t, rx_t  # extend to 2 elements

        # TODO: enable variable rates during a single TR
        self._rx_divs = np.round(np.array(rx_t) * fpga_clk_freq_MHz).astype(np.uint32)
        self._rx_ts = self._rx_divs / fpga_clk_freq_MHz

        if not hasattr(rx_lo, "__len__"):
            rx_lo = rx_lo, rx_lo  # extend to 2 elements
        self._rx_lo = rx_lo

        assert grad_board in ("ocra1", "gpa-fhdo"), "Unknown gradient board!"
        if grad_board == "ocra1":
            gradb_class = gb.OCRA1
            self._gpa_fhdo_offset_time = 0
        else:
            gradb_class = gb.GPAFHDO
            self._gpa_fhdo_offset_time = gpa_fhdo_offset_time
        self.gradb = gradb_class(self.server_command, grad_max_update_rate)
        self._grad_latency = grad_latency

        if initial_wait is None:
            # auto-set the initial wait to be long enough for initial gradient configuration to finish, plus 1us for miscellaneous startup
            self._initial_wait = 1 + 1 / grad_max_update_rate

        self._auto_leds = auto_leds

        self._mimo_master = mimo_master
        if self._mimo_master:
            self._trig_output_time = trig_output_time
            self._slave_trig_latency = slave_trig_latency
        else:
            self._trig_output_time = 0
            self._slave_trig_latency = 0

        if trig_timeout < 0:
            self._trig_wait_time = np.int32(-1)
        else:
            self._trig_wait_time = np.round(trig_timeout * fpga_clk_freq_MHz).astype(np.uint32)

        assert (seq_csv is None) or (
            seq_dict is None
        ), "Cannot supply both a sequence dictionary and a CSV file."
        self._csv = None
        self._seq = None
        if seq_dict is not None:
            self.add_flodict(seq_dict)
        elif seq_csv is not None:
            self._csv = seq_csv  # None unless a CSV was supplied

        self._print_infos = print_infos
        self._assert_errors = assert_errors

        if init_gpa:
            self.gradb.init_hw()

        if halt_and_reset:
            halted = sc.command({"halt_and_reset": 0}, self._s)[0][4]["halt_and_reset"]
            assert (
                halted
            ), "Could not halt the execution of an existing sequence. Please file a bug report."

        self._fix_cic_scale = fix_cic_scale
        self._set_cic_shift = set_cic_shift
        self._flush_old_rx = flush_old_rx
        self._allow_user_init_cfg = allow_user_init_cfg

    def __del__(self):
        if self._close_socket:
            self._s.close()

    def server_command(self, server_dict):
        return sc.command(server_dict, self._s, self._print_infos, self._assert_errors)

    def get_rx_ts(self):
        return self._rx_ts

    def set_lo_freq(self, lo_freq):
        # lo_freq: either a single floating-point value, or an iterable of up to three values for each marga NCO

        # extend lo_freq to 3 elements
        if not hasattr(lo_freq, "__len__"):
            lo_freq = lo_freq, lo_freq, lo_freq  # extend to 3 elements
        elif len(lo_freq) < 3:
            lo_freq = lo_freq[0], lo_freq[1], lo_freq[0]  # extend from 2 to 3 elements

        # calculate real LO freqs -- TODO: print for debugging
        self._dds_phase_steps = np.round(
            2**31 / self._fpga_clk_freq_MHz * np.array(lo_freq)
        ).astype(np.uint32)
        self._lo_freqs = self._dds_phase_steps * self._fpga_clk_freq_MHz / (2**31)

        # force recompilation -- TODO: could just modify the instructions setting the LO
        self._seq_compiled = False

    def flo2int(self, seq_dict):
        """Convert a floating-point sequence dictionary to an integer binary
        dictionary"""
        intdict = {}

        ## Various functions to handle the conversion
        def times_us(farr):
            """farr: float array, times in us units; [0, inf)"""
            # negative values will get rejected at a later stage
            return np.round(self._fpga_clk_freq_MHz * farr).astype(np.int64)

        # TODO: alter calculation later to use these integer offsets, to avoid floating-point addition
        initial_wait_b = times_us(self._initial_wait)
        grad_latency_b = times_us(self._grad_latency)
        gpa_fhdo_offset_time_b = times_us(self._gpa_fhdo_offset_time)

        # TODO add initial_wait_b to this, so that it doesn't need to occur later
        master_offset_b = times_us(self._trig_output_time + self._slave_trig_latency)

        def tx_real(farr):
            """farr: float array, [-1, 1]"""
            return np.round(32767 * farr).astype(np.uint16)

        def tx_complex(times, farr, tolerance=2e-6):
            """times: float time array, farr: complex float array, [-1-1j, 1+1j]
            tolerance: minimum difference two values need to be considered binary-unique (2e-6 corresponds to ~19 bits)
            -- returns a tuple with repeated elements removed"""
            idata, qdata = farr.real, farr.imag
            unique = lambda k: np.concatenate([[True], np.abs(np.diff(k)) > tolerance])
            # DEBUGGING: use the below lambda instead to avoid stripping repeated values
            # unique = lambda k: np.ones_like(k, dtype=bool)
            idata_u, qdata_u = unique(idata), unique(qdata)
            tbins = (
                times_us(times[idata_u] + self._initial_wait) + master_offset_b,
                times_us(times[qdata_u] + self._initial_wait) + master_offset_b,
            )
            txbins = (tx_real(idata[idata_u]), tx_real(qdata[qdata_u]))
            return tbins, txbins

        for key, (times, vals) in seq_dict.items():
            # each possible dictionary entry returns a tuple (even if one element) for the binary dictionary to send to marcompile
            tbin = None  # will be overwritten later
            if key in ["tx0_i", "tx0_q", "tx1_i", "tx1_q"]:
                valbin = (tx_real(vals),)
                keybin = (key,)
            elif key in ["tx0", "tx1"]:
                tbin, valbin = tx_complex(times, vals)
                keybin = key + "_i", key + "_q"
            elif key in [
                "grad_vx",
                "grad_vy",
                "grad_vz",
                "grad_vz2",
                "fhdo_vx",
                "fhdo_vy",
                "fhdo_vz",
                "fhdo_vz2",
                "ocra1_vx",
                "ocra1_vy",
                "ocra1_vz",
                "ocra1_vz2",
            ]:
                # marcompile will figure out whether the key matches the selected grad board
                keyb, channel = self.gradb.key_convert(key)
                keybin = (keyb,)  # tuple
                valbin = (self.gradb.float2bin(vals, channel),)

                # hack to de-synchronise GPA-FHDO outputs, to give the
                # user the illusion of being able to output on several
                # channels in parallel
                if self._gpa_fhdo_offset_time:
                    # TODO the floating-point additions here are unfavourable, should pre-calculate per-channel offsets!
                    tbin = (times_us(times + self.initial_wait) + channel * gpa_fhdo_offset_time_b - grad_latency_b + master_offset_b,)
                else:
                    # compensate latency
                    tbin = (times_us(times + self._initial_wait) - grad_latency_b + master_offset_b,) # + initial_wait_b - grad_latency_b,)

            elif key in ["rx0_rate", "rx1_rate"]:
                keybin = (key,)
                valbin = (vals.astype(np.uint16),)
            elif key in [
                "rx0_rate_valid",
                "rx1_rate_valid",
                "rx0_rst_n",
                "rx1_rst_n",
                "rx0_en",
                "rx1_en",
                "tx_gate",
                "rx_gate",
                "trig_out",
            ]:
                keybin = (key,)
                # binary-valued data
                valbin = (vals.astype(np.int32),)
                for vb in valbin:
                    assert np.all(
                        (0 <= vb) & (vb <= 1)
                    ), "Binary columns must be [0,1] or [False, True] valued"
            elif key in ["leds"]:
                keybin = (key,)
                valbin = (vals.astype(np.int32),)
            elif key in ["lo0_freq", "lo1_freq", "lo2_freq"]:
                keybin = (key,)
                valbin = (
                    np.round(2**31 / self._fpga_clk_freq_MHz * vals).astype(np.uint32),
                )
            else:
                warnings.warn("Unknown marga experiment dictionary key: " + key)
                continue

            # default tbin
            if tbin is None:
                tbin = (times_us(times + self._initial_wait) + master_offset_b,)

            for t, k, v in zip(tbin, keybin, valbin):
                intdict[k] = (t, v)

        return intdict

    def add_intdict(self, seq_intdict, append=True):
        """Add an integer-format dictionary to the sequence, or replace the old (time, value) tuples with new ones"""
        if self._seq is None:
            self._seq = {}

        for name, sb in seq_intdict.items():
            if name in self._seq.keys() and append:
                a, b = self._seq[name]
                self._seq[name] = (np.append(a, sb[0]), np.append(b, sb[1]))
            else:
                self._seq[name] = sb

    def add_flodict(self, flodict, append=True):
        """Add a floating-point dictionary to the sequence"""
        assert (
            self._csv is None
        ), "Cannot replace the dictionary for an Device class created from a CSV"
        self.add_intdict(self.flo2int(flodict), append)
        self._seq_compiled = False

    def compile(self):
        """Convert either dictionary or CSV file into machine code, with
        extra machine code at the start to ensure the system is initialised to
        the correct state.

        Initially, configure the RX rates and set the LEDs.
        Remainder of the sequence will be as programmed.
        """

        # RX and LO configuration
        tstart = 50  # cycles before doing anything
        rx_wait = 50  # cycles to run RX before setting rate, then later resetting again
        initial_cfg = {  # 'rx0_rst_n': ( np.array([tstart, tstart + 2*rx_wait]), np.array([1, 0]) ),
            # 'rx1_rst_n': ( np.array([tstart, tstart + 2*rx_wait]), np.array([1, 0]) ),
            "rx0_rst_n": (np.array([tstart]), np.array([1])),
            "rx1_rst_n": (np.array([tstart]), np.array([1])),
            "lo0_freq": (np.array([tstart]), np.array([self._dds_phase_steps[0]])),
            "lo1_freq": (np.array([tstart]), np.array([self._dds_phase_steps[1]])),
            "lo2_freq": (np.array([tstart]), np.array([self._dds_phase_steps[2]])),
            "lo0_rst": (np.array([tstart, tstart + 1]), np.array([1, 0])),
            "lo1_rst": (np.array([tstart, tstart + 1]), np.array([1, 0])),
            "lo2_rst": (np.array([tstart, tstart + 1]), np.array([1, 0])),
        }

        # Set CIC decimation rate and internal shift, if necessary, and calculate CIC scale correction
        rx0_words, self._rx0_cic_factor = mc.cic_words(
            self._rx_divs[0], self._set_cic_shift
        )
        rx1_words, self._rx1_cic_factor = mc.cic_words(
            self._rx_divs[1], self._set_cic_shift
        )
        if not self._fix_cic_scale:  # clear the correction factor
            self._rx0_cic_factor = 1
            self._rx1_cic_factor = 1
        rx0r_st = tstart + rx_wait
        wds = len(rx0_words)
        ar0 = np.arange(wds, dtype=int)
        ar1 = np.arange(wds + 1, dtype=int)
        ow = np.ones(wds + 1, dtype=int)
        ow[-1] = 0

        initial_cfg.update(
            {
                "rx0_rate": (tstart + rx_wait + ar0, np.array(rx0_words)),
                "rx1_rate": (tstart + rx_wait + ar0, np.array(rx1_words)),
                "rx0_rate_valid": (tstart + rx_wait + ar1, ow),
                "rx1_rate_valid": (tstart + rx_wait + ar1, ow),
            }
        )

        # LO source configuration (only set non-default values if necessary)
        if self._rx_lo[0] != 0:
            initial_cfg.update(
                {"rx0_lo": (np.array([tstart]), np.array([self._rx_lo[0]]))}
            )
        if self._rx_lo[1] != 0:
            initial_cfg.update(
                {"rx1_lo": (np.array([tstart]), np.array([self._rx_lo[1]]))}
            )

        # Automatic trigger pulse if master
        if self._mimo_master:
            # trigger has to occur before the slave trigger latency is applied
            # to all the regular channels during compilation
            self.add_flodict({'trig_out': (np.array([-self._slave_trig_latency, -self._slave_trig_latency + 1]), np.array([1, 0]))})

        # Automatic LED scan
        if self._auto_leds:
            led_steps = 256

            # find max time used in system
            ultimate_time = 0
            for k in self._seq:
                loc_last_time = self._seq[k][0][-1]
                if loc_last_time > ultimate_time:
                    ultimate_time = loc_last_time

            if ultimate_time < 256:
                led_steps = ultimate_time
            led_times = np.linspace(tstart, ultimate_time + tstart, 256).astype(
                np.int64
            )
            led_vals = np.linspace(1, 256, led_steps).astype(np.uint32)
            initial_cfg.update(
                {"leds": (led_times, led_vals)}
            )  # should overlap with any previous LED settings

        # do not clear relevant dictionary values if user-defined configuration
        # of init parameters at runtime is allowed
        self.add_intdict(initial_cfg, append=self._allow_user_init_cfg)

        mc.grad_board = self._grad_board  # BAD, SETTING GLOBAL VARIABLE - TODO FIX
        if self._csv:  # TODO: handle CSVs in Device class. UNTESTED, SHOULD WORK
            self._machine_code = np.array(
                mc.csv2bin(
                    self._csv,
                    self.gradb.bin_config["initial_bufs"],
                    # TODO: can add extra manipulation here, e.g. add to another array etc
                    self.gradb.bin_config["latencies"],
                    self._trig_wait_time,
                ),
                dtype=np.uint32,
            )
        else:
            self._machine_code = np.array(
                mc.dict2bin(
                    self._seq,
                    self.gradb.bin_config["initial_bufs"],
                    # TODO: can add extra manipulation here, e.g. add to another array etc
                    self.gradb.bin_config["latencies"],
                    self._trig_wait_time,
                ),
                dtype=np.uint32,
            )

        self._seq_compiled = True

    def get_flodict(self, intd=None):
        """Calculate floating-point dictionaries based on the data inside the
        Device class so far -- useful for plotting or testing the sequence"""

        if intd is None:
            if not self._seq_compiled:
                self.compile()

            intd = self._seq

        flodict = {}

        def convert_t(t_bin, y):
            # add a zero event in the beginning, and shift the times to the 'user frame'
            t = (
                np.concatenate(([0], t_bin)) / self._fpga_clk_freq_MHz
                - self._initial_wait
            )
            # add a zero value in the beginning of outputs
            y2 = np.concatenate(([0], y))
            return t, y2

        # Convert TX channels
        for txl in ["tx0_i", "tx0_q", "tx1_i", "tx1_q"]:
            try:
                t_bin, tx_bin = intd[txl]
                t, tx = convert_t(t_bin, tx_bin.astype(np.int16) / 32768)
                flodict[txl] = (t, tx)
            except KeyError:
                continue

        # Convert gradient channels
        for gradl in self.gradb.keys():
            try:
                t_bin, grad_bin = intd[gradl]
                t, grad = convert_t(t_bin, self.gradb.bin2float(grad_bin))
                flodict[gradl] = (t, grad)
            except KeyError:
                continue

        # Convert RX enable channels
        for rxl in ["rx0_en", "rx1_en"]:
            try:
                t_bin, rx = intd[rxl]
                t, rx = convert_t(t_bin, rx)
                flodict[rxl] = (t, rx)
            except KeyError:
                continue

        # Convert digital outputs
        for iol in ["tx_gate", "rx_gate", "trig_out", "leds"]:
            try:
                t_bin, io = intd[iol]
                t, io = convert_t(t_bin, io)
                if iol == "leds":
                    io = io.astype(np.uint8).astype(float) / 256
                    flodict[iol] = (t, io)
            except KeyError:
                continue

        return flodict

    def plot_sequence(self, axes=None):
        """axes: 4-element tuple of axes upon which the TX, gradients, RX and digital I/O plots will be drawn.
        If not provided, plot_sequence() will create its own."""
        if axes is None:
            _, axes = plt.subplots(4, 1, figsize=(12, 8), sharex="col")

        (txs, grads, rxs, ios) = axes

        fd = self.get_flodict()

        # Plot TX channels
        for txl in ["tx0_i", "tx0_q", "tx1_i", "tx1_q"]:
            try:
                txs.step(*fd[txl], where="post", label=txl)
            except KeyError:
                continue

        # Plot gradient channels
        for gradl in self.gradb.keys():
            try:
                grads.step(*fd[gradl], where="post", label=gradl)
            except KeyError:
                continue

        # Plot RX enable channels
        for rxl in ["rx0_en", "rx1_en"]:
            try:
                rxs.step(*fd[rxl], where="post", label=rxl)
            except KeyError:
                continue

        # Plot digital outputs
        for iol in ["tx_gate", "rx_gate", "trig_out", "leds"]:
            try:
                ios.step(*fd[iol], where="post", label=iol)
            except KeyError:
                continue

        for ax in axes:
            ax.legend()
            ax.grid(True)

        ios.set_xlabel(r"time ($\mu$s)")
        return fd

    def run(self):
        """compile the TX and grad data, send everything over.
        Returns the resultant data"""

        if not self._seq_compiled:
            self.compile()

        if self._flush_old_rx:
            rx_data_old, _ = sc.command({"read_rx": 0}, self._s)
            # TODO: do something with RX data previously collected by the server

        rx_data, msgs = sc.command({"run_seq": self._machine_code.tobytes()}, self._s)

        rxd = rx_data[4]["run_seq"]
        rxd_iq = {}

        # (1 << 24) just for the int->float conversion to be reasonable - exact value doesn't matter for now
        rx0_norm_factor = self._rx0_cic_factor / (1 << 24)
        rx1_norm_factor = self._rx0_cic_factor / (1 << 24)

        try:
            rxd_iq["rx0"] = rx0_norm_factor * (
                np.array(rxd["rx0_i"]).astype(np.int32).astype(float)
                + 1j * np.array(rxd["rx0_q"]).astype(np.int32).astype(float)
            )
        except (KeyError, TypeError):
            pass

        try:
            rxd_iq["rx1"] = rx1_norm_factor * (
                np.array(rxd["rx1_i"]).astype(np.int32).astype(float)
                + 1j * np.array(rxd["rx1_q"]).astype(np.int32).astype(float)
            )
        except (KeyError, TypeError):
            pass

        return rxd_iq, msgs

    def close_server(self, only_if_sim=False):
        ## Either always close server, or only close server if it's a simulation
        if (
            not only_if_sim
            or sc.command({"are_you_real": 0}, self._s)[0][4]["are_you_real"]
            == "simulation"
        ):
            sc.send_packet(
                sc.construct_packet({}, 0, command=sc.close_server_pkt), self._s
            )


def test_rx_scaling(
    lo_freq=0.5,
    rf_amp=0.5,
    rf_steps=True,
    rx_time=50,
    rx_periods=[600],
    rx_padding=20,
    plot_rx=False,
):
    dev = Device(
        lo_freq=lo_freq,
        rx_t=rx_periods[0] / lc.fpga_clk_freq_MHz,
        fix_cic_scale=False,
        set_cic_shift=False,
        allow_user_init_cfg=True,
        flush_old_rx=True,
    )
    tr_t = 0
    tr_period = rx_time + rx_padding
    rx_lengths = []

    def single_pulse_tr(tstart, rx_period=20):
        if rf_steps:
            tx_shape = (
                np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2, 0]) * rf_amp
            )
            tx_times = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) * (
                rx_time - 2 * rx_padding
            )
        else:
            tx_shape = np.array([1, 0]) * rf_amp
            tx_times = np.array([0, 1]) * (rx_time - 2 * rx_padding)
        tx_seq = (tstart + 5 + rx_padding + tx_times, tx_shape)
        rx_seq = (tstart + 5 + np.array([0, rx_time]), np.array([1, 0]))

        # Rate adjustment
        set_cic_shift = dev._set_cic_shift
        rx_words, _ = mc.cic_words(rx_period, set_cic_shift)
        rx_wait = 100
        rxr_st = tstart + rx_wait
        wds = len(rx_words)
        ar0 = np.arange(wds, dtype=int)
        ar1 = np.arange(wds + 1, dtype=int)
        ow = np.ones(wds + 1, dtype=int)
        ow[-1] = 0
        rate_seq = (tstart + (rx_wait + ar0) / lc.fpga_clk_freq_MHz, np.array(rx_words))
        rate_en_seq = (tstart + (rx_wait + ar1) / lc.fpga_clk_freq_MHz, ow)

        value_dict = {
            "tx0": tx_seq,
            "tx1": tx_seq,
            "rx0_en": rx_seq,
            "rx1_en": rx_seq,
            "rx0_rate": rate_seq,
            "rx1_rate": rate_seq,
            "rx0_rate_valid": rate_en_seq,
            "rx1_rate_valid": rate_en_seq,
        }

        # if rx_period % 2:
        #     value_dict.pop('rx0_rate_valid', None)
        #     value_dict.pop('rx1_rate_valid', None)

        return value_dict

    for rt in rx_periods:
        dev.add_flodict(single_pulse_tr(tr_t, rt))
        tr_t += tr_period
        rx_lengths.append(int(rx_time * lc.fpga_clk_freq_MHz / rt) + 1)

    rxd, msgs = dev.run()
    dev.close_server(True)

    # rx_lengths_a = np.array(rx_lengths)
    # print("RX lengths from calculation: ", rx_lengths)

    pulse_means = []

    if plot_rx:
        rx0_mag = np.abs(rxd["rx0"])
        rx1_mag = np.abs(rxd["rx1"])

        initial_excess = 5  # amount to throw away from the start of each acquisition
        M = 0

        plt.figure(figsize=(12, 6))
        plt.xlabel("RX sample")
        plt.ylabel("RF magnitude")
        for k, l in zip(rx_periods, rx_lengths):
            x = np.arange(M + initial_excess, M + l)
            y = np.array(rx1_mag[M + initial_excess : M + l])

            _, cic_correction = mc.cic_words(k)
            ysc = y * cic_correction

            # avoid off-by-1 errors - bit of a hack
            if len(ysc) != len(x):
                x = x[:-1]

            if True:
                plt.plot(x, ysc)
            M += l
            # pulse_means.append( y[y > 0.5*y.max()].mean() )
            # pulse_means.append( y.mean() )
            pulse_means.append(ysc.max())

        plt.figure(figsize=(8, 7))
        pma = np.array(pulse_means)
        plt.subplot(2, 1, 1)
        plt.plot(rx_periods, pma / pma.mean())
        plt.xlabel("RX decimation")
        plt.ylabel("RX peak amp")
        plt.grid(True)
        # plt.subplot(3,1,2)
        # plt.plot(rx_periods, rx_lengths)
        # plt.legend(['rx samples'])
        # plt.xlabel('RX decimation')
        # plt.grid(True)
        plt.subplot(2, 1, 2)
        _, cic_corrections = mc.cic_words(rx_periods)
        plt.plot(rx_periods, cic_corrections)
        plt.xlabel("RX decimation")
        plt.ylabel("RX CIC correction factor")
        plt.grid(True)
        plt.show()


def test_gpa_calibration():
    dev = Device(init_gpa=True)

    # test calibration on GPA-FHDO capable of driving a full current load
    # if false, just test over a narrow range
    full_current = True

    if full_current:
        dev.gradb.calibrate(
            channels=[0],
            max_current=0.7,
            num_calibration_points=30,
            averages=5,
            settle_time=0.005,
            poly_degree=5,
        )
        dev.gradb.calibrate(
            channels=[0],
            max_current=0.7,
            num_calibration_points=30,
            averages=1,
            test_cal=True,
        )
    else:
        dev.gradb.calibrate(
            channels=[0],
            max_current=0.05,
            num_calibration_points=30,
            averages=5,
            settle_time=0.005,
            poly_degree=2,
        )
        dev.gradb.calibrate(
            channels=[0],
            max_current=0.05,
            num_calibration_points=30,
            averages=1,
            test_cal=True,
        )


def test_lo_change():
    dev = Device(auto_leds=False)
    dev.add_flodict({"tx0": (np.array([1]), np.array([0.5]))})
    dev.compile()
    dev.set_lo_freq(2)
    dev.compile()
    dev.run()
    # dev.close_server(only_if_sim=True)


if __name__ == "__main__":
    print("No tests are run.")
    if False:
        test_rx_scaling(
            lo_freq=0.5,
            rf_amp=1,
            rf_steps=False,
            rx_time=300,
            rx_padding=10,
            # rx_periods = np.ones(10, dtype=int)*30,
            # rx_periods=np.arange(4, 400, 1),
            # rx_periods=np.arange(4, 40, 1),
            rx_periods=np.arange(4, 100, 1),
            # rx_periods=np.arange(10, 400, 1),
            # rx_periods=np.array([5, 10, 20, 50, 100, 200, 500]),
            plot_rx=False,
        )

    if False:
        test_gpa_calibration()

    if False:
        test_lo_change()
