# coding: utf-8
import os
import sys
import configparser
conf = configparser.ConfigParser()
conf.read("E:\PySpace\Amadeus\server\\server_conf.ini")


def load_model_path():
    return conf.get("model", "model_path")


def load_amadeus_ip_port():
    ip = conf.get("amadeus_server", "ip")
    port = conf.get("amadeus_server", "port")
    return ip, int(port)


def load_amadeus_recv_buffer_size():
    return int(conf.get("amadeus_server", "recv_buffer_size"))


def load_annoy_ip_port():
    ip = conf.get("annoy_server", "ip")
    port = conf.get("annoy_server", "port")
    return ip, int(port)


def load_annoy_recv_buffer_size():
    return int(conf.get("annoy_server", "recv_buffer_size"))


def load_wx_batch():
    return int(conf.get("wx_server", "batch_size"))


def load_wx_wait():
    return int(conf.get("wx_server", "wait_time"))


def load_wx_wait_unit():
    return int(conf.get("wx_server", "wait_unit"))
