import logging
import logging.handlers
import pickle
import struct
import SocketServer
import argparse
import os


class LogRecordStreamHandler(SocketServer.StreamRequestHandler):
    def handle(self):
        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break
            slen = struct.unpack('>L', chunk)[0]
            chunk = self.connection.recv(slen)
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))
            obj = pickle.loads(chunk)
            print(obj)


class LogRecordSocketReceiver(SocketServer.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, host='localhost',
                 port=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 handler=LogRecordStreamHandler):
        SocketServer.ThreadingTCPServer.__init__(self, (host, port), handler)


def parse_args():
    parser = argparse.ArgumentParser(description='Tcp listener for receive log')
    parser.add_argument('--host', default='localhost',
                        type=str, help='host for log server')
    parser.add_argument('--port',
                        default=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                        type=int,
                        help='port for log server')
    return parser.parse_args()


def main():
    args = parse_args()
    print('About to start TCP server...')
    print('Process Id: {}'.format(os.getpid()))
    server = LogRecordSocketReceiver(host=args.host, port=args.port, handler=LogRecordStreamHandler)
    server.serve_forever()


if __name__ == '__main__':
    main()
