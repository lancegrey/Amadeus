import itchat
import threading
import time
import socket
import json
import random
from server import load_conf


class WxServer(object):
    def __init__(self):
        self.debug = False
        self.silent = False
        self.response_sig = threading.Semaphore(1)
        self.replying = False
        self.collection = []
        self.wait_time = load_conf.load_wx_wait()
        self.s_for_wait = self.wait_time
        self.wait_unit = load_conf.load_wx_wait_unit()
        self.batch_size = load_conf.load_wx_batch()
        # trick, but useful
        self.collection_func = lambda msg: self.text_collection(msg)
        itchat.msg_register(itchat.content.TEXT)(self.collection_func)

    def text_collection(self, msg):
        self.response_sig.acquire()
        if msg.fromUserName == "filehelper":
            self.control(msg["text"])
        if self.silent:
            self.response_sig.release()
            return
        text = msg['Text']
        pre_ask_amadeus = json.loads(self.search(text))
        values = []
        if pre_ask_amadeus["type"] == "annoy":
            if len(pre_ask_amadeus["value"]) > 0:
                if float(pre_ask_amadeus["value"][1]) < 0.8:
                    values = pre_ask_amadeus["value"][2]
        else:
            values = pre_ask_amadeus["value"]
        if len(values) >= 1:
            choose = random.choice(values)
            print(values)
            print(choose)
            if self.debug:
                out_type = pre_ask_amadeus["type"]
                choose = choose + "\ndebug-info: " + out_type
            msg.user.send(choose)
        else:
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
            data.append(msg['Text'])
        if len(data) > 0:
            data = json.dumps(data)
            ret = self.ask(data)
            ret = json.loads(ret)
        else:
            ret = []
        for msg, ans in zip(self.collection, ret):
            ans = "\n".join(ans) + "\ndebug-info: generation"
            msg.user.send(ans)
        self.collection = []

    def ask(self, data):
        host, port = load_conf.load_amadeus_ip_port()
        buffer_size = load_conf.load_amadeus_recv_buffer_size()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        sock.sendall(data.encode())
        return sock.recv(buffer_size).decode()

    def search(self, text):
        host, port = load_conf.load_annoy_ip_port()
        buffer_size = load_conf.load_annoy_recv_buffer_size()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        sock.sendall(text.encode())
        return sock.recv(buffer_size).decode()

    def control(self, text):
        if text == "system#debug":
            self.debug = True
        elif text == "system#normal":
            self.debug = False
        elif text == "system#silent":
            self.silent = True


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
