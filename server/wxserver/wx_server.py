import itchat
import threading
import time
import socket
import json
import datetime
from server import load_conf


class WxServer(object):
    def __init__(self):
        self.response_sig = threading.Semaphore(1)
        self.replying = False
        self.collection = []
        self.wait_time = load_conf.load_wx_wait()
        self.s_for_wait = self.wait_time
        self.wait_unit = load_conf.load_wx_wait_unit()
        self.batch_size = load_conf.load_wx_batch()
        self.amadeus_ip, self.amadeus_port = load_conf.load_amadeus_ip_port()
        self.buffer_size = load_conf.load_amadeus_recv_buffer_size()
        # trick, but useful
        self.collection_func = lambda msg: self.text_collection(msg)
        itchat.msg_register(itchat.content.TEXT)(self.collection_func)

    def text_collection(self, msg):
        self.response_sig.acquire()
        self.collection.append(msg)
        if len(self.collection) >= self.batch_size:
            self.s_for_wait = self.wait_time
            self.response()
        self.response_sig.release()

    def wait(self):
        self.response_sig.acquire()
        if self.s_for_wait <= 0:
            self.response()
            self.s_for_wait = self.wait_time
        self.response_sig.release()
        time.sleep(self.wait_unit)
        self.response_sig.acquire()
        self.s_for_wait -= self.wait_unit
        self.response_sig.release()

    def response(self):
        data = []
        for msg in self.collection:
            if msg['FromUserName'] != "":
                data.append(msg['Text'])
        if len(data) > 0:
            data = json.dumps(data)
            ret = self.ask(data)
            ret = json.loads(ret)
        else:
            ret = []
        for msg, ans in zip(self.collection, ret):
            ans = "\n".join(ans)
            msg.user.send(ans)
        self.collection = []

    def ask(self, data):
        host, port = self.amadeus_ip, server.amadeus_port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        sock.sendall(data.encode())
        return sock.recv(self.buffer_size).decode()


class WxWait(threading.Thread):
    def __init__(self, server):
        threading.Thread.__init__(self)
        self.server = server

    def run(self):
        while True:
            self.server.wait()


if __name__ == "__main__":
    server = WxServer()
    waiter = WxWait(server)
    waiter.start()
    itchat.auto_login()
    # itchat.send('Hello, filehelper', toUserName='filehelper')
    itchat.run()
