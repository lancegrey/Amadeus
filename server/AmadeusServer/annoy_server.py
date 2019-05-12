# coding: utf-8
import json
import socket
from server import load_conf
from Amadeus.brain.talk import ann
from Amadeus.brain.talk import Logic_rape


class AnnoyServer(ann.AnnoySearch):
    def __init__(self):
        super(AnnoyServer, self).__init__()
        self.logic_rape = Logic_rape.LogicRape()
        self.ip, self.port = load_conf.load_annoy_ip_port()
        self.recv_buffer_size = load_conf.load_annoy_recv_buffer_size()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._bind()

    def _bind(self):
        self.sock.bind((self.ip, self.port))

    def _handle_request(self, connection):
        buf = connection.recv(self.recv_buffer_size)
        #try:
        qus = buf.decode()
        ans = {"type": "logic", "value": []}
        ans["value"] = self.logic_rape.search(qus)
        if len(ans["value"]) <= 0:
            ans["value"] = self.search(qus)
            ans["type"] = "annoy"
        ret_buf = json.dumps(ans)
        #except Exception as e:
        #    ret_buf = json.dumps({"error": "1"})
        #    print(e)
        connection.sendall(ret_buf.encode())

    def service(self):
        self.sock.listen(1)
        print("suc service!")
        while True:
            connection, address = self.sock.accept()
            self._handle_request(connection)
            connection.close()


if __name__ == "__main__":
    annoy_server = AnnoyServer()
    annoy_server.service()
