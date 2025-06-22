/**@file iface.hpp
   @brief Communications interface to the host via Ethernet

   General comment about error handling: for functions that will only
   ever be called remotely by the client, it is easier to call
   server_action's add_warning() and add_error() methods. However for
   more universal functions, such as those used to set up the
   hardware initially, it's better to throw runtime errors that will
   be caught by the next higher-level functions available.

   The mpack error callback functions should be used to return
   standard error packets via the interface, with only the message
   'MPack error found'.
*/

#ifndef _IFACE_HPP_
#define _IFACE_HPP_

extern "C" {
#include "server_config.h"
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
}

#include "mpack/mpack.h"
#include <string>
#include <vector>
#include <stdexcept>

class hardware;

/// @brief handle properties of the stream, like the file descriptor and whether
/// (TODO: add more if necessary)
struct stream_t {
	int fd;
};

struct mpack_tree_t;
struct mpack_node_t;

/// @brief stream read and write functions
size_t read_stream(mpack_tree_t* tree, char* buffer, size_t count);
void write_stream(mpack_writer_t* writer, const char* buffer, size_t count);

/// @brief MaRCoS msgpack packet types
enum marcos_packet {
	marcos_request=0,
	marcos_emergency_stop=1,
	marcos_close_server=2,
	marcos_reply=128,
	marcos_reply_error=129
};

enum command_return {
	c_ok=0,
	c_err=-1,
	c_warn=-2
};

/// @brief Various kinds of MaRCoS-specifc errors
struct marcos_error: public std::runtime_error {
	marcos_error(const char *msg) : runtime_error(msg) {}
	marcos_error(const std::string &msg) : runtime_error(msg) {}
};

struct hw_error: public marcos_error {
	hw_error(const char *msg) : marcos_error(msg) {}
	hw_error(const std::string &msg) : marcos_error(msg) {}
};

struct data_error: public marcos_error {
	data_error(const char *msg) : marcos_error(msg) {}
	data_error(const std::string &msg) : marcos_error(msg) {}
};

struct mpack_error: public std::runtime_error {
	mpack_error(const char *msg) : runtime_error(msg) {}
	mpack_error(const std::string &msg) : runtime_error(msg) {}
};

/// @brief Server action class, encapsulating the logic for telling
/// the hardware what to do and internally constructing a reply
/// containing relevant data etc returned from the hardware. Also
/// includes errors, warnings and/or info from each stage of the process.
class server_action {
public:
	/// @brief Interpret the incoming request and start preparing the reply in advance
	server_action(mpack_node_t request_root, mpack_writer_t* writer); // TODO: add hardware object
	~server_action();
	/// @brief Wrapper to provide mpack nodes to the hardware. The
	/// nodes should be the command argument (usually maps); see
	/// the MaRCoS interface specification (TODO: wiki URL here)
	/// for more information. command_present returns 1 if the
	/// command was found, 0 otherwise; -1 if there was an mpack
	/// error.
	mpack_node_t get_command_and_start_reply(const char* cstr, int &command_present);
	/// @brief Return the number of commands requested by the client; negative if there's an error (TODO)
	size_t command_count();
	/// @brief Getter for the writer object
	mpack_writer_t* get_writer() {return _wr;}
	/// @brief Run the request on the hardware, returning a basic error status
	int process_request();
	/// @brief Finish filling the buffer to reply: include the status messages and anything else left over. Return: TODO
	ssize_t finish_reply();
	/// @brief Flush reply buffer to the stream
	void send_reply();
	void add_error(std::string s);
	void add_warning(std::string s);
	void add_info(std::string s);
	/// @brief True if the reader tree has had any errors
	bool reader_err();
private:
	/// @brief Short for request data; payload containing request data from client specifying what it wants the server to do
	mpack_node_t _rd;
	mpack_writer_t* _wr;
	unsigned _request_type, _reply_index, _request_version;
	std::vector<std::string> _errors, _warnings, _infos;

	/// @brief Encode the vectors of strings containing messages
	/// (errors, warnings and infos) into msgpack
	void encode_messages();

	/// @brief Verify the version of the protocol used by the client.
	/// If the major version differs, throw a runtime error.
	/// If the minor version differs, throw a warning.
	/// If the debug version differs, send back info.
	void check_version();

	/// @brief Carry out emergency stop: zero the RF and DACs, halt sequence, etc...
	void emergency_stop();
};

///@brief Interface manager class, encapsulating the interface logic
class iface {
public:
	iface(unsigned port=11111);

	/// @brief Set up socket
	void init(unsigned port=11111);
	/// @brief Run request-response loop
	void run_stream(); // main
	/// @brief Unpack and act on each received packet, calling
	/// other methods and passing in nodes of the MPack tree where
	/// necessary. Return the size used in reply_buf.
	int process_message(mpack_node_t root, char *reply_buf);
	~iface();
private:
	bool _run_iface = true;

	struct sockaddr_in _address;
	size_t _addrlen;

	int _server_fd;
	stream_t _stream_fd;
};

#endif
