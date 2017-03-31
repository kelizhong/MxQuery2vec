import socket


def local_ip():
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except:
        ip = "127.0.0.1"
    return ip