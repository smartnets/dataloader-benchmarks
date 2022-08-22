import socket


def ping_udp(epoch, iter, pos):
    byte_message = bytes(f"{epoch}-{iter}-{pos}", "utf-8")
    opened_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    opened_socket.sendto(byte_message, ("8.8.8.8", 80))
