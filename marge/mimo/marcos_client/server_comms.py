#!/usr/bin/env python3

import msgpack, warnings

from marmachine import MarServerWarning

version_major = 1
version_minor = 0
version_debug = 6
version_full = (version_major << 16) | (version_minor << 8) | version_debug

request_pkt = 0
emergency_stop_pkt = 1
close_server_pkt = 2
reply_pkt = 128


def construct_packet(
    data,
    packet_idx=0,
    command=request_pkt,
    version=(version_major, version_minor, version_debug),
):
    vma, vmi, vd = version
    assert vma < 256 and vmi < 256 and vd < 256, "Version is too high for a byte!"
    version = (vma << 16) | (vmi << 8) | vd
    fields = [command, packet_idx, 0, version, data]
    return fields


def send_packet(packet, socket):
    socket.sendall(msgpack.packb(packet))

    unpacker = msgpack.Unpacker()
    packet_done = False
    while not packet_done:
        buf = socket.recv(1024)
        if not buf:
            break
        unpacker.feed(buf)
        for o in unpacker:  # ugly way of doing it
            return o  # quit function after 1st reply (could make this a thread in the future)


def command(server_dict, socket, print_infos=False, assert_errors=False):
    packet = construct_packet(server_dict)
    reply = send_packet(packet, socket)
    return_status = reply[5]

    if print_infos and "infos" in return_status:
        print("Server info:")
        for k in return_status["infos"]:
            print(k)

    if "warnings" in return_status:
        for k in return_status["warnings"]:
            warnings.warn(k, MarServerWarning)

    if "errors" in return_status:
        if assert_errors:
            assert "errors" not in return_status, return_status["errors"][0]
        else:
            for k in return_status["errors"]:
                warnings.warn("SERVER ERROR: " + k, RuntimeWarning)

    return reply, return_status
