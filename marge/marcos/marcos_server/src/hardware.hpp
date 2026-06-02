/** @file hardware.hpp
    @brief Hardware management and interface/data transfer to the Zynq PL.
*/

#ifndef _HARDWARE_HPP_
#define _HARDWARE_HPP_

#include <inttypes.h> // TODO is this the right include?
#include <unistd.h>
#include <vector>

// Memory-mapped device sizes
static const unsigned PAGESIZE = sysconf(_SC_PAGESIZE); // should be 4096 (4KiB) on both x86_64 and ARM
static const unsigned SLCR_SIZE = PAGESIZE,
	MARGA_SIZE = 128*PAGESIZE,
	MARGA_MEM_SIZE = 64*PAGESIZE;
static const unsigned MARGA_MEM_MASK = 0x3ffff;
static const unsigned MARGA_RX_FIFO_SPACE = 16384;

// marga internal states
static const unsigned MAR_STATE_IDLE = 0, MAR_STATE_PREPARE = 1, MAR_STATE_RUN = 2,
	MAR_STATE_COUNTDOWN = 3, MAR_STATE_TRIG = 4, MAR_STATE_TRIG_FOREVER = 5,
	MAR_STATE_HALT = 8;

static const unsigned MAR_BUFS = 24;
static const unsigned MAR_BUF_ALL_EMPTY = (1 << MAR_BUFS) - 1;

static const unsigned MAR_STATUS_GPA_MASK = 0x00030000;

struct mpack_node_t;

class server_action;

class hardware {
public:
	hardware();
	~hardware();

	int run_request(server_action &sa);

        /// @brief Set up shared memory, control registers etc; these
        /// aspects are not client-configurable. If compiled on x86,
        /// just mimics the shared memory.
	void init_mem();

	/// @brief Halt the FSM, interrupting any ongoing sequence and/or readout in progress
	void halt();

	/// @brief Halt and reset all outputs to default values, even
	/// if the cores are currently running. Activated when an
	/// emergency stop command arrives.
	void halt_and_reset();
private:
	// Config variables
	unsigned _pc_tries_limit = 1000000; // how long to wait if PC isn't changing
	unsigned _idle_tries_limit = 1000000; // how long to wait for the end of the sequence if memory is fully written
	unsigned _read_tries_limit = 1000; // retry attempts for each data sample
	unsigned _halt_tries_limit = 1000000; // read retry attemps for HALT state at the end of the sequence
	unsigned _gpa_idle_tries_limit = 1000; // how long to wait for GPA interfaces to become idle at the end of a sequence
	unsigned _samples_per_halt_check = 2; // how often to check halt status (in read samples) during normal readout
	unsigned _min_rx_reads_per_loop = 16;
	unsigned _max_rx_reads_per_loop = 1024;

	// Peripheral register addresses in PL
	volatile uint32_t *_slcr, *_mar_base, *_ctrl, *_direct, *_exec, *_status,
		*_status_latch, *_buf_err, *_buf_full, *_buf_empty, *_rx_locs,
		*_rx0_i_data, *_rx1_i_data, *_rx0_q_data, *_rx1_q_data;

	volatile char *_mar_mem;

	/// @brief Write

	/// @brief Read out RX FIFOs into vectors. Reads out either
	/// all the data available (default), or just the number of
	/// reads specified by max_reads. Doesn't strictly respect
	/// max_reads, only within ~4 reads. Returns the amount of 32b
	/// data in the most-filled FIFO, *before* reading began.
	unsigned read_rx(std::vector<uint32_t> &rx0_i, std::vector<uint32_t> &rx0_q,
	                 std::vector<uint32_t> &rx1_i, std::vector<uint32_t> &rx1_q,
	                 const unsigned max_reads = 100000);

	// methods to support simulation; most efficient to inline them
	inline void wr32(volatile uint32_t *addr, uint32_t data);
	inline uint32_t rd32(volatile uint32_t *addr);
	void* hw_memcpy(volatile void *s1, const void *s2, size_t n);
	size_t hw_mpack_node_copy_data(mpack_node_t node, volatile char *buffer, size_t bufsize);
};

#endif
