import warnings
warnings.simplefilter('ignore')
from struct import *
import time
import os
import sqlalchemy
import mysql.connector
# from collections import namedtuple
import numpy as np
import pandas as pd
from numpy.fft import fft,rfft, irfft, fftfreq, ifft
from scipy.signal import butter, lfilter, freqz, find_peaks, lfilter, peak_widths, peak_prominences, convolve, gaussian
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.integrate import quad
from array import array
from numpy import random, arange
from numpy.fft import rfft, rfftfreq
from scipy.signal import sosfilt, sosfreqz
import datetime as dt
import yaml



# обьявление необходимых функций
# фильтрация сигналов
def butter_bandpass(lowcut, highcut, fs, order=7):

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=7):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


#  Создание сокета с базой данных
def connect_database(host_name, user_name, user_password, database_name):
    host_name = host_name
    user_name = user_name
    user_password = user_password
    database_name = database_name

    try:
        engine = sqlalchemy.create_engine(
            'mysql+mysqlconnector://' + user_name + ':' + user_password + '@' + host_name + '/' + database_name)
        connection = engine.connect()

    except Exception as e:
        return (False, 0, e)
    else:
        e = "Sucsess connection."
        connection.close()
        return (True, engine, e)


# обьявление функции парсинга даты-времени из метки в изоформат
def timestamp_to_unix(timestamp):
    import datetime as dt
    msek = (timestamp - 20210101000000) % 1
    base_time = '2021-01-01 00:00:00'
    time_stamp = []
    for elm in str(timestamp):
        time_stamp.append(elm)
    timestamp = time_stamp
    year = timestamp[0] + timestamp[1] + timestamp[2] + timestamp[3]
    month = timestamp[4] + timestamp[5]
    day = timestamp[6] + timestamp[7]
    hour = timestamp[8] + timestamp[9]
    minitue = timestamp[10] + timestamp[11]
    seconds = timestamp[12] + timestamp[13]
    iso_timestamp = year + "-" + month + "-" + day + " " + hour + ":" + minitue + ":" + seconds
    unix_timestamp = (dt.datetime.fromisoformat(iso_timestamp)).timestamp()
    unix_base_time = (dt.datetime.fromisoformat(base_time)).timestamp()
    unix_timestamp = unix_timestamp - unix_base_time + msek
    return unix_timestamp


# обьявление функции парсинга файла конфигурации в единый словарь
def read_config(config_yaml_file):

    summ_config_dict = {}
    with open(config_yaml_file, 'r') as file:
        config = yaml.full_load(file)
        for i in range(0, len(config)):
            for key, value in config[i].items():
                #             print(key, ":", value)
                summ_config_dict[key] = value
    return (summ_config_dict)
