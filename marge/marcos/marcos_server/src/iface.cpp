#include "mpack/mpack.h"
#include "version.hpp"
#include "iface.hpp"
#include "hardware.hpp"
#include <sstream>
#include <iostream>
#include <cassert>

extern hardware *hw;

std::string version_str(unsigned ver) {
	std::stringstream v;
	v << ((ver & 0xff0000) >> 16) << "." << ((ver & 0xff00) >> 8) << "." << (ver & 0xff);
	return v.str();
}

size_t read_stream(mpack_tree_t* tree, char* buffer, size_t count) {
	stream_t* stream = (stream_t *)mpack_tree_context(tree);
	ssize_t step = read(stream->fd, buffer, count);
	if (step <= 0) mpack_tree_flag_error(tree, mpack_error_io);
	return step;
}

void write_stream(mpack_writer_t* writer, const char* buffer, size_t count) {
	stream_t* stream = (stream_t *)mpack_writer_context(writer);
	ssize_t amount = write(stream->fd, buffer, count);
	if (amount <= 0) mpack_writer_flag_error(writer, mpack_error_io);
//	return amount;
}

server_action::server_action(mpack_node_t request_root, mpack_writer_t* writer):
// server_action::server_action(mpack_node_t request_root, char *reply_buffer):
//	_reply_buffer(reply_buffer)
	_wr(writer)
{
	auto r = request_root;
	_request_type = mpack_node_uint(mpack_node_array_at(r, 0));
	_reply_index = mpack_node_uint(mpack_node_array_at(r, 1));
	_request_version = mpack_node_uint(mpack_node_array_at(r, 3));

	_rd = mpack_node_array_at(r, 4);
	check_version();

	// Start constructing the reply
	mpack_start_array(_wr, 6);

	if (_request_type == marcos_emergency_stop) mpack_write_u32(_wr, marcos_reply_error);
//	else if (_request_type == marcos_close_server) mpack_write_u32(_wr,
	else mpack_write_u32(_wr, marcos_reply);

	mpack_write_u32(_wr, _reply_index + 1); // reply index for the client to keep track
	mpack_write_u32(_wr, 0); // unused for now
	mpack_write_u32(_wr, SERVER_VERSION_UINT); // protocol version
}

server_action::~server_action() {
	// if (mpack_tree_destroy(&_tree) != mpack_ok) {
	// 	perror("Error occurred tearing down the reply MPack tree\n");
	// 	// throw std::runtime_error("Error occurred tearing down an MPack tree"); // SHOULD NOT THROW EXCEPTIONS IN DESTRUCTORS
	// }
}

size_t server_action::command_count() {
	return mpack_node_map_count(_rd);
}

mpack_node_t server_action::get_command_and_start_reply(const char* cstr, int &status) {
	mpack_node_t node = mpack_node_map_cstr_optional(_rd, cstr);
	if (mpack_node_is_missing(node)) status = 0;
	else if (mpack_node_is_nil(node)) status = -1;
	else {
		// write the string into the reply node; next write
		// will presumably be the reply from the hardware
		// object
		mpack_write_cstr(_wr, cstr);
		status = 1;
	}
	return node;
}

int server_action::process_request() {
	if (_request_type != marcos_request) {
		mpack_start_map(_wr, 0); // dummy map, has no content since no results will be returned
		mpack_finish_map(_wr);

		if (_request_type == marcos_emergency_stop) {
			// emergency stop should be processed immediately before anything else!
			emergency_stop();
			add_error("Emergency stop not yet implemented!");
			add_warning("Tried to carry out emergency stop!");
			return 0;
		} else if (_request_type == marcos_close_server) {
			add_info("Shutting down server.");
			return -1; // message to iface to shut down server (maybe more in the future)
		}
	}
	// default: run the request on the hardware, feeding in this server_action object
	try {
		// int problems = hw->run_request(*this);
		hw->run_request(*this);
	} catch (std::runtime_error &e) {
		add_error(e.what());
	}

	return 0;
}

ssize_t server_action::finish_reply() {
	encode_messages();
	// Do other stuff potentially in the future if the protocol expands, like status updates etc
	mpack_finish_array(_wr);
	return 0;
}

void server_action::send_reply() {
	mpack_writer_flush_message(_wr);
}

bool server_action::reader_err() {
	assert(false && "Not yet implemented");
	return false;
}

void server_action::encode_messages() {
	// Assumes that we're on the final element of the MaRCoS reply
	bool send_errors = _errors.size(), send_warnings = _warnings.size(), send_infos = _infos.size(); // nonzero = true
	int map_size = send_errors + send_warnings + send_infos; // how many nonzero
	mpack_start_map(_wr, map_size);

	// Should use a generic function, but too lazy
	if (send_errors) {
		mpack_write_cstr(_wr, "errors");
		mpack_start_array(_wr, _errors.size());
		for (auto &k: _errors) mpack_write_cstr(_wr, k.c_str());
		mpack_finish_array(_wr);
	}

	if (send_warnings) {
		mpack_write_cstr(_wr, "warnings");
		mpack_start_array(_wr, _warnings.size());
		for (auto &k: _warnings) mpack_write_cstr(_wr, k.c_str());
		mpack_finish_array(_wr);
	}

	if (send_infos) {
		mpack_write_cstr(_wr, "infos");
		mpack_start_array(_wr, _infos.size());
		for (auto &k: _infos) mpack_write_cstr(_wr, k.c_str());
		mpack_finish_array(_wr);
	}
	mpack_finish_map(_wr);
}

void server_action::add_error(std::string s) { _errors.push_back(s); }
void server_action::add_warning(std::string s) { _warnings.push_back(s); }
void server_action::add_info(std::string s) { _infos.push_back(s); }

void server_action::check_version() {
	char client_version_major = (_request_version & 0xff0000) >> 16;
	char client_version_minor = (_request_version & 0xff00) >> 8;
	char client_version_debug = _request_version & 0xff;

	std::stringstream e;

	if (client_version_major != VERSION_MAJOR) { // major version mismatch
		e << "Client version " << version_str(_request_version) << " significantly different from server version " << SERVER_VERSION_STR;
		add_error(e.str());
		// throw std::runtime_error(e.str());
	} else if (client_version_minor != VERSION_MINOR) {
		e << "Client version " << version_str(_request_version) << " different from server version " << SERVER_VERSION_STR;
		add_warning(e.str());
	} else if (client_version_debug != VERSION_DEBUG) {
		e << "Client version " << version_str(_request_version) << " differs slightly from server version " << SERVER_VERSION_STR;
		add_info(e.str());
	}
}

void server_action::emergency_stop() {
	// perror("Emergency stop not yet implemented - do this soon!");
	fprintf(stderr, "Emergency stop not yet implemented - do this soon!\n");
	// exit(EXIT_FAILURE);
}

iface::iface(unsigned port) {
	init(port);
}

void iface::init(unsigned port) {
	if ( (_server_fd = socket(AF_INET, SOCK_STREAM, 0) ) < 0) {
		perror("socket failed");
		exit(EXIT_FAILURE);
	}

	int reuseaddr = 1; // whether to reuse the address
	if (setsockopt(_server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
	               (void *)&reuseaddr , sizeof(reuseaddr)) ) {
		perror("setsockopt failed");
		exit(EXIT_FAILURE);
	}

	_address.sin_family = AF_INET;
	_address.sin_addr.s_addr = INADDR_ANY;
	_address.sin_port = htons(port);
	_addrlen = sizeof(_address);

	if (bind(_server_fd, (struct sockaddr *)&_address, _addrlen) < 0) {
		perror("bind failed");
		exit(EXIT_FAILURE);
	}

	if ( listen(_server_fd, 10) ) {
		perror("listen failed");
		exit(EXIT_FAILURE);
	}
}

void iface::run_stream() {
	const unsigned max_size = 1024*1024*32;
	const unsigned max_nodes = 8192;

	char *reply_buf = (char *)malloc(max_size);

	while (_run_iface) {
		// block until a client connects
		if((_stream_fd.fd = accept(_server_fd, NULL, NULL)) < 0) {
			// if((my_socket = accept(stream_fd.fd, (struct sockaddr *)&address, (socklen_t *)&addrlen)) < 0) {
			fprintf(stderr, "[%s line %d] socket accept failed\n", __FILE__, __LINE__);
			exit(EXIT_FAILURE);
		}

		// mpack reader mechanism
		mpack_tree_t tree;
		mpack_tree_init_stream(&tree, &read_stream, &_stream_fd, max_size, max_nodes);

		// mpack writer mechanism
		// CONTINUE HERE; use mpack_writer_init(), mpack_writer_set_context(), mpack_writer_set_flush() and maybe mpack_writer_set_error_handler(); then when ready to send stuff, mpack_writer_flush_message(). First though, test basic string replies.
		mpack_writer_t writer;
		mpack_writer_init(&writer, reply_buf, max_size);
		mpack_writer_set_context(&writer, &_stream_fd);
		mpack_writer_set_flush(&writer, &write_stream);

		while (true) {
			mpack_tree_parse(&tree); // blocking
			//mpack_tree_try_parse(&tree); // non-blocking, for future use
			mpack_error_t err = mpack_tree_error(&tree);
			if ( err != mpack_ok) {
				if (err != mpack_error_io) {
					// mpack_error_io thrown whenever the client disconnects.
					// Haven't found a way around this so just ignoring it for now.
					// Below will catch other kinds of error though.
					fprintf(stderr, "Parsing the MPack tree failed: %s\n", mpack_error_to_string(err));
				}
				break;
			}

			// Reply: use a constant buffer
			try {
				server_action sa(mpack_tree_root(&tree), &writer);
				int sa_status = sa.process_request(); // run hardware operations or whatever else is needed
				if (sa_status != 0) _run_iface = false; // shut down server gracefully
				sa.finish_reply();
				sa.send_reply();
			} catch (mpack_error &) {
				fprintf(stderr, "MPack error!\n");
				// TODO: implement an error callback
				// for the relevant MPack functions,
				// and return a packet stating that
				// there was an MPack error
			}
		}

		mpack_error_t err = mpack_tree_destroy(&tree);
		if (err != mpack_ok and err != mpack_error_io) {
			// As for the streaming, mpack_error_io always happens at the moment
			fprintf(stderr, "Tearing down the MPack tree failed: %s\n", mpack_error_to_string(err));
		}

		mpack_error_t err2 = mpack_writer_destroy(&writer);
		if (err2 != mpack_ok) fprintf(stderr, "Destroying the MPack writer failed: %s\n", mpack_error_to_string(err2));

		if (close(_stream_fd.fd) < 0) {
			fprintf(stderr, "[%s line %d] closing socket failed\n", __FILE__, __LINE__);
			exit(EXIT_FAILURE);
		}
	}

	free(reply_buf);
}

iface::~iface() {
}
