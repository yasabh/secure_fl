# this file is taken from the python client example data are not part of the distribution downloaded by the install script
# file available here: https://github.com/data61/MP-SPDZ/tree/master/ExternalIO
# it is slightly modified due to a bug in the example

import socket, ssl
import struct
import time

class Client:
    def __init__(self, hostnames, port_base, my_client_id):
        # ctx = ssl.SSLContext()
        ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        name = 'C%d' % my_client_id
        prefix = 'Player-Data/%s' % name
        ctx.load_cert_chain(certfile=prefix + '.pem', keyfile=prefix + '.key')
        ctx.load_verify_locations(capath='Player-Data')

        self.sockets = []
        for i, hostname in enumerate(hostnames):
            for j in range(10000):
                try:
                    plain_socket = socket.create_connection(
                        (hostname, port_base + i))
                    break
                except ConnectionRefusedError:
                    if j < 60:
                        time.sleep(1)
                    else:
                        raise
            octetStream(b'%d' % my_client_id).Send(plain_socket)
            self.sockets.append(ctx.wrap_socket(plain_socket,
                                                server_hostname='P%d' % i))

        self.specification = octetStream()
        self.specification.Receive(self.sockets[0])

    def receive_triples(self, T, n):
        #triples = [[0, 0, 0] * n]
        triples = [[0, 0, 0] for i in range(n)]
        
        os = octetStream()
        for socket in self.sockets:
            os.Receive(socket)
            for triple in triples:
                for i in range(3):
                    t = T()
                    t.unpack(os)
                    triple[i] += t
        res = []
        for i, triple in enumerate(triples):
            prod = triple[0] * triple[1]
            if prod != triple[2]:
                raise Exception(
                    'invalid triple %i, diff %s' % (i, hex(prod.v - triple[2].v)))
        return triples

    def send_private_inputs(self, values):
        T = type(values[0])
        triples = self.receive_triples(T, len(values))
        os = octetStream()
        for value, triple in zip(values, triples):
            (value + triple[0]).pack(os)
        for socket in self.sockets:
            os.Send(socket)

    def receive_outputs(self, T, n):
        triples = self.receive_triples(T, n)
        return [triple[0] for triple in triples]

class octetStream:
    def __init__(self, value=None):
        self.buf = b''
        self.ptr = 0
        if value is not None:
            self.buf += value

    def reset_write_head(self):
        self.buf = b''
        self.ptr = 0

    def Send(self, socket):
        socket.send(struct.pack('<i', len(self.buf)))
        socket.send(self.buf)

    def Receive(self, socket):
        length = struct.unpack('<I', socket.recv(4))[0]
        self.buf = socket.recv(length)
        self.ptr = 0

    def store(self, value):
        self.buf += struct.pack('<i', value)

    def get_int(self, length):
        buf = self.buf[self.ptr:self.ptr + length]
        self.ptr += length
        if length == 4:
            return struct.unpack('<i', buf)[0]
        elif length == 8:
            return struct.unpack('<q', buf)[0]
        raise ValueError()

    def get_bigint(self):
        sign = self.consume(1)[0]
        assert(sign in (0, 1))
        length = self.get_int(4)
        if length:
            res = 0
            buf = self.consume(length)
            for i, b in enumerate(reversed(buf)):
                res += b << (i * 8)
            if sign:
                res *= -1
            return res
        else:
            return 0

    def consume(self, length):
        self.ptr += length
        return self.buf[self.ptr - length:self.ptr]

