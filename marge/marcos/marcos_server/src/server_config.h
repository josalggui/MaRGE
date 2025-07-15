#ifndef _SERVER_CONFIG_H_
#define _SERVER_CONFIG_H_

#include <stdint.h>

// Memory map
#ifdef __arm__
static const uint32_t SLCR_OFFSET = 0xf8000000,
	MARGA_OFFSET = 0x43c00000, // ocra_grad_ctrl core's main offset (TODO: fill this in with Vivado value once it's ready)
	MARGA_MEM_OFFSET = MARGA_OFFSET + 262144; // 256KiB offset inside the core to access the BRAMs

#else // emulate the memory in a single file on the desktop, for the purposes of debugging, emulation etc
static const unsigned EMU_PAGESIZE = 0x1000; // 4 KiB, page size on the RP and my desktop machine
static const uint32_t SLCR_OFFSET = 0, // 1 page in size
	MARGA_OFFSET = SLCR_OFFSET + EMU_PAGESIZE,
	MARGA_MEM_OFFSET = MARGA_OFFSET + 64 * EMU_PAGESIZE,
	END_OFFSET = MARGA_MEM_OFFSET + 64 * EMU_PAGESIZE;
#endif

// Auxiliary parameters
static const unsigned COMMS_BUFFER = 8192; // TODO expand if necessary

#endif
