import sys
import os
import netifaces as ni
import socket
from os import getenv
from typing import List
from pydantic import BaseModel

from src.models.acquisition import AppMode


def get_ip_mac():  
    ni.ifaddresses('en0')
    return str(ni.ifaddresses('en0')[ni.AF_INET][0]['addr'])


def get_ip(device="R", wifi_interface="eth0"): # eth0, wlan0
    try: 
        try:
            host_name = socket.getfqdn() 
        except Exception as e: 
            print(e)
            host_name = os.system("hostname")
        if device == "R":
            host_ip = str(ni.ifaddresses(wifi_interface)[ni.AF_INET][0]['addr'])
        else:
            host_ip = socket.gethostbyname(host_name)   # wind
    except Exception as e: print(e)
    return str(host_ip)


def get_host_name_ip():
    if sys.platform.startswith('linux'):  # raspberry
        return get_ip("R")
    elif sys.platform.startswith('win32'):  # windows
        return get_ip("W")
    elif sys.platform.startswith('darwin'):  # mac
        return get_ip_mac()


class Config(BaseModel):
    ENV: str = getenv("ENV", "dev")
    VERSION: str = getenv("VERSION", "0.0.1")
    HOST_IP: str = get_host_name_ip()
    DATABASE_URL: str = f"sqlite://{getenv('DATABASE_PATH', '/data/db.sqlite')}"
    PORT: int = int(getenv("SERVER_PORT", "8020"))
    MOVIE: str = getenv("MOVIE_PATH", "None")
    MOVIES_FOLDER: str = getenv("MOVIES_FOLDER", "./static/movies")
    AUTO: bool = getenv("AUTO", "0") == "1"
    LOG: bool = getenv("LOG", "0") == "1"
    PORTS: List[int] = list(map(int, getenv('DEVICE_PORTS', "8801, 8802, 8803, 8804, 8805, 8806, 8807, 8808, 8809, 8828").split(',')))
    SAMPLING_RATE: int = int(getenv('SAMPLING_RATE', 100))
    CHANNELS: List[int] = list(map(int, getenv('CHANNELS', "1,2,3,4,5,6,7,8").split(',')))
    SEND_DATA_CHANNELS: List[int] = list(map(int, getenv('CHANNELS', "1, 3").split(',')))
    MODE: AppMode = AppMode.stop
    START_TIME: int = 0
    SECRET_KEY: str = "LGgBcCHhvfZOGbBGILSV"
    CURRENT_SESSION_ID: int = -1 
    SEGMENTATION: str = "EDA"
    EDA_CHANNEL: str = 'ai_1'


config = Config()
