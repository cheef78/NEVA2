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
from user_functions import *

class Neva2():


    def __init__(self, args):
        self.neva_name = args[0]
        self.dat_dir = args[1]
        self.unpak_dir = args[2]
        self.slots_config = args[3]
        self.dpk_config = args[4]
        self.config_algoritm = args[5]
        self.config_criteria = args[6]
        self.database_config = args[7]
        self.train_id = args[8][0]
        self.date_time = args[8][1]
        self.file_full_name = args[8][2]

    def info(self):
        print(
            "Для этой системы НЕВА-2 установлены следующие настройки: \n        - Имя системы: {0}\n        - Папка хранения dat-файлов: {1}\n        - Папка хранения обработанных файлов: {2}\n".format(
                self.neva_name, self.dat_dir, self.unpak_dir))
        print()
        print("Конфигурация слотов системы:")

        for i in (self.slots_config.keys()):
            print()
            print('Сторонность слотов: ', i)
            print()
            for k in range(len(self.slots_config.get(i))):
                for j in range(len((self.slots_config.get(i))[k])):
                    text = ('Серийный номер слота: ', 'Число активных каналов в слоте, шт: ',
                            'Частота выборки данных в слоте, Гц: ', 'Масштабные коэффициенты для каналов в слоте: ')
                    print(text[j] + str(((self.slots_config.get(i))[k])[j]))
                if len(((self.slots_config.get(i))[k])[3]) != int(((self.slots_config.get(i))[k])[1]):
                    print("Число активных каналов в слоте и число масштабных коэффициентов не совпадает!!")
                print()

    def unpack(self, file_to_csv='False'):

        train_id = self.train_id
        flag1 = file_to_csv

        # print (dat_files)
        sigs_all = pd.DataFrame(columns=['value', 'sensor', 'time', 'side'])
        # obrabotka pravoy storoni
        for i in (self.slots_config.keys()):
            j = 0
            for k in range(len(self.slots_config.get(i))):
                side = i
                slot_serial = ((self.slots_config.get(i))[k])[0]
                channels = ((self.slots_config.get(i))[k])[1]
                slot_rate = ((self.slots_config.get(i))[k])[2]
                use_time_stamp = ((self.slots_config.get(i))[k])[3]
                gain_coeff = ((self.slots_config.get(i))[k])[4]
                #                 print (slot_serial,channels,slot_rate)
                dat_file = self.file_full_name + slot_serial + ".dat"
                #                 print (self.file_full_name)
                #                 print (dat_file)
                with open(dat_file, 'rb') as f:
                    data = f.read()
                    signal_isxod = []
                    for k in range(0, len(data), 8):
                        signal_isxod.append(*unpack('=d', data[k:k + 8]))
                    if use_time_stamp == 0:
                        signal = []
                        for item in signal_isxod:
                            if item < 10000:
                                signal.append(item)
                    else:
                        signal = []
                        for z in range(0, len(signal_isxod)):
                            if z <= 3:
                                signal.append(signal_isxod[z])
                            else:
                                if signal_isxod[z] < 10000:
                                    signal.append(signal_isxod[z])

                for l in range(channels):
                    sigs = pd.DataFrame()
                    sig = np.array([signal[m] for m in range(0 + l, len(signal), channels)])
                    sigs['value'] = sig
                    sigs['sensor'] = int(j)
                    sigs['time'] = sigs.index / slot_rate
                    sigs['side'] = str(side)

                    #    Блок уравнивания тиков на метки времени
                    if use_time_stamp == 0:
                        sigs['value'] = sigs['value'] * int(gain_coeff[l])

                    else:

                        #                         print (sigs['value'][0])

                        curr_unix_time = timestamp_to_unix(sigs['value'][0])
                        #                         print (curr_unix_time)
                        sigs['time'] = sigs['time'] + curr_unix_time
                        sigs['value'][0] = sigs['value'][1]
                        #                         print (int(gain_coeff[l]))
                        sigs['value'] = sigs['value'] * int(gain_coeff[l])

                    j = j + 1
                    sigs_all = pd.concat([sigs_all, sigs], ignore_index=True)

                    # Приведение исходных сигналов к 0 уровню
        sensorlist = pd.unique(sigs_all['sensor'])
        sensorlist.sort()
        sidelist = pd.unique(sigs_all['side'])
        sidelist.sort()
        time_step_for_zero_leveling = 0.5
        for side in sidelist:
            for sensor in sensorlist:
                channel_time_start = sigs_all['time'].loc[
                    (sigs_all['sensor'] == sensor) & (sigs_all['side'] == side)].min()
                channel_time_zero_levelenig_end = channel_time_start + time_step_for_zero_leveling
                zero_level = (sigs_all['value'].loc[(sigs_all['sensor'] == sensor) & (sigs_all['side'] == side) & (
                            sigs_all['time'] >= channel_time_start) & (sigs_all[
                                                                           'time'] <= channel_time_zero_levelenig_end)]).mean()
                sigs_all['value'].loc[(sigs_all['sensor'] == sensor) & (sigs_all['side'] == side)] = \
                sigs_all['value'].loc[(sigs_all['sensor'] == sensor) & (sigs_all['side'] == side)] - zero_level

        if flag1 == "True":
            csv_name = self.unpak_dir + "\\" + self.neva_name + "_" + self.date + self.time.replace(":",
                                                                                                    "-") + "_all_data_from_all_slots.csv"
            sigs_all.to_csv(csv_name, index=False, sep=';')

        return sigs_all

    def find_dpk_wheel_time(self, dpk_sigs):

        # funkciya opredeleniya momenta vremeni proxoda kolesa cherez DPK
        # dpk_sigs - massiv signalov s DPK, privedennyi k nulyu po porogam nastroiki DPK
        sigs = dpk_sigs
        sigs.reset_index(inplace=True)
        # print (sigs)
        borders = []
        peaks = []
        for i in range(sigs['value'].count() - 1):
            if ((sigs['value'][i] == 0 and sigs['value'][i + 1] > 0) or (
                    sigs['value'][i] > 0 and sigs['value'][i + 1] == 0)):
                borders.append((sigs['time'][i]))
        tik = ((sigs['time'][(sigs['time'].count() - 1)]) - sigs['time'][0]) / (sigs['time'].count())
        for i in range(int(len(borders) / 2)):
            peaks.append((borders[2 * i] + borders[2 * i + 1] + tik) / 2)
        print("Число осей, шт.  = " + str(len(peaks)))
        return peaks

    def find_wheel_time_speed(self, signal_data_freim, file_to_csv='False'):

        signal = signal_data_freim
        flag1 = file_to_csv
        dpk_config = self.dpk_config

        axels = pd.DataFrame(columns=['speed', 'numb_axel', 'sensor', 'time', 'side'])

        #         массивы и данные для разбивки осей по вагонам
        between_axels = pd.DataFrame(
            columns=['side', 'dist_between', 'axel_start', 'axel_end', 'wagon_count', 'wagon_type',
                     'AxlesAmount_InCar'])

        l_base_4axel_wagon = 1.85
        l_base_under_4axel_vagon = 6.85
        l_bases_under_4axel_vagons = (5.650, 8.150, 6.800, 6.800, 5.950, 6.800, 5.950, 8.650, 10.390, 17.150, 5.950)
        l_couple_4axel_wagon = 1.81
        l_couples_4axel_wagons = (1.795, 1.870, 2.280, 2.310, 1.598, 2.270, 1.455, 1.795, 2.050, 2.795, 2.565)

        l_base_sochlenen_wagon = 1.85
        l_base_under_sochlenen_vagon = 5.55
        l_couple_sochlenen_wagon = 1.20

        l_base_6axel_wagon = 1.5
        l_base_under_6axel_vagon = 5.85
        l_base_between_axel_6axel_vagon = 5.85

        l_base_8axel_wagon = 1.5
        l_base_under_8axel_vagon = 5.85
        l_base_between_bogie_8axel_vagon = 1.35

        max_koeff = 1.1
        min_koeff = 0.9

        for i in (dpk_config.keys()):
            ch_i = (dpk_config.get(i))[0]
            ch_ii = (dpk_config.get(i))[1]
            levl_ch_i = (dpk_config.get(i))[2]
            levl_ch_ii = (dpk_config.get(i))[3]
            l_dpk12 = (dpk_config.get(i))[4]
            l_dpk1_tvs1 = (dpk_config.get(i))[5]
            l_dpk1_tvs2 = (dpk_config.get(i))[6]
            l_dpk1_tvs3 = (dpk_config.get(i))[7]
            l_dpk1_tvs4 = (dpk_config.get(i))[8]
            l_dpk1_tvs5 = (dpk_config.get(i))[9]
            l_dpk1_tvs6 = (dpk_config.get(i))[10]
            side = str(i)

            signal_df = signal.copy()

            # obrabotka storoni
            sigs_i = signal_df.loc[((signal_df['sensor'] == int(ch_i)) & (signal_df['side'] == side))]
            sigs_ii = signal_df.loc[((signal_df['sensor'] == int(ch_ii)) & (signal_df['side'] == side))]
            #             sigs_i.to_csv("sigs_i_befor_loc_" + side + ".csv", index=False, sep = ';')
            #             sigs_ii.to_csv("sigs_ii_befor_loc_" + side + ".csv", index=False, sep = ';')
            sigs_i.loc[signal_df['value'] < levl_ch_i, 'value'] = 0
            sigs_ii.loc[signal_df['value'] < levl_ch_ii, 'value'] = 0
            # print (sigs_i)
            # sigs_i.to_csv("sigs_i_after_loc_" + side + ".csv", index=False, sep = ';')
            # print (sigs_ii)
            # sigs_ii.to_csv("sigs_ii_after_loc_" + side + ".csv", index=False, sep = ';')
            if sigs_ii['value'].count() == 0 or sigs_i['value'].count() == 0 or (
                    sigs_ii['value'].count() != sigs_ii['value'].count()):
                print(
                    "Сторона: " + side + ". \n Данные сигналов с ДПК не валидны. Дальнейшая обработка записи невозможна!!")
                return (False)
            times_i = np.array(Neva2.find_dpk_wheel_time(self, sigs_i))
            times_ii = np.array(Neva2.find_dpk_wheel_time(self, sigs_ii))
            #             Вывод числа меток прохода осей
            #             print (len(times_i))
            #             print (len(times_ii))
            if len(times_ii) == 0 or len(times_i) == 0 or (len(times_ii) != len(times_i)):
                print(
                    "Сторона: " + side + ". \n Данные массивов времени прохода осей через ДПК не валидны. Дальнейшая обработка записи невозможна!!")
                return (False)
                #                 axels = pd.DataFrame(columns=['speed', 'numb_axel', 'sensor', 'time', 'side'])
            #                 return axels

            #             times_ii = times_ii + 0.5 # "эту строку обязательно убрать при работе для реальных данных!!!!."

            speed = l_dpk12 / (times_ii - times_i)

            #            Разброс осей по порядковому номеру вагона и номеру оси в вагоне
            # определение межосевых расстояний
            delta_axels = []
            axel_start = []
            axel_end = []

            for i in range(len(times_i) - 1):
                delta_axels_i = abs((times_i[i + 1] - times_i[i]) * speed[i])
                delta_axels_ii = abs((times_ii[i + 1] - times_ii[i]) * speed[i])
                axel_start.append(i)
                axel_end.append(i + 1)
                delta_axels.append((delta_axels_i + delta_axels_ii) / 2)
            between_axels_side = pd.DataFrame(
                columns=['side', 'dist_between', 'axel_start', 'axel_end', 'wagon_count', 'wagon_type',
                         'AxlesAmount_InCar'])
            between_axels_side['dist_between'] = delta_axels
            between_axels_side['axel_start'] = axel_start
            between_axels_side['axel_end'] = axel_end
            between_axels_side['side'] = side

            between_axels = pd.concat([between_axels, between_axels_side], ignore_index=True)
            #             between_axels.to_csv("between_axels_all.csv", index=False, sep = ';', encoding="utf-8")
            #             csv_name = self.unpak_dir + "\\" + self.neva_name + "_" + self.date + self.time.replace(":", "-") + "_distance_between_accels_all.csv"
            #             between_axels_side.to_csv(csv_name, index=False, sep = ';', encoding="utf-8")

            base_4axel_indexes = between_axels['dist_between'].loc[
                (between_axels['side'] == side) & (between_axels['dist_between'] >= min_koeff * l_base_4axel_wagon) & (
                            between_axels['dist_between'] <= max_koeff * l_base_4axel_wagon)].index

            base_6axel_indexes = between_axels['dist_between'].loc[
                (between_axels['side'] == side) & (between_axels['dist_between'] >= min_koeff * l_base_6axel_wagon) & (
                            between_axels['dist_between'] <= max_koeff * l_base_6axel_wagon)].index

            base_8axel_indexes = between_axels['dist_between'].loc[(between_axels['side'] == side) & (
                        between_axels['dist_between'] >= min_koeff * l_base_between_bogie_8axel_vagon) & (between_axels[
                                                                                                              'dist_between'] <= max_koeff * l_base_between_bogie_8axel_vagon)].index

            #         l_base_4axel_wagon = 1.85
            #         l_base_under_4axel_vagon = 5.85
            #         l_couple_4axel_wagon = 1.61

            #         l_base_sochlenen_wagon = 1.85
            #         l_base_under_sochlenen_vagon = 5.55
            #         l_couple_sochlenen_wagon = 1.20

            #         l_base_6axel_wagon = 1.5
            #         l_base_under_6axel_vagon = 5.85
            #         l_base_between_axel_6axel_vagon = 5.85

            #         l_base_8axel_wagon = 1.5
            #         l_base_under_8axel_vagon = 5.85
            #         l_base_between_bogie_8axel_vagon = 1.35

            #         max_koeff = 1.1
            #         min_koeff = 0.9

            #             выводчисла тележек
            #             print ("Сторона системы для анализа = ", side)
            #             print ("Число тележек в 4-х осных и сочлененных вагонах = ", len(base_4axel_indexes))
            #             print ("Число тележек в 6-и осных вагонах = ",len(base_6axel_indexes)/2)
            #             print ("Число тележек в 8-и осных вагонах = ",len(base_8axel_indexes))
            #             wagon_count = 0
            #             i = 0
            #             if len(base_6axel_indexes) == 0 and len(base_8axel_indexes) == 0:
            #                 while i <= (len(base_4axel_indexes)-1):
            #                     print (wagon_count, i, base_4axel_indexes[i])
            #                     dist_0 = between_axels['dist_between'].loc[base_4axel_indexes[i]]
            #                     dist_1 = between_axels['dist_between'].loc[base_4axel_indexes[i]+1]
            #                     dist_2 = between_axels['dist_between'].loc[base_4axel_indexes[i]+2]
            #                     dist_3 = between_axels['dist_between'].loc[base_4axel_indexes[i]+3]
            #                     dist_4 = between_axels['dist_between'].loc[base_4axel_indexes[i]+4]
            #                     dist_5 = between_axels['dist_between'].loc[base_4axel_indexes[i]+5]

            #                     print (dist_0, dist_1, dist_2, dist_3, dist_4, dist_5)

            #                     if  (min_koeff*l_base_4axel_wagon <=  dist_0 <= max_koeff*l_base_4axel_wagon)\
            #                     and (min_koeff*l_base_under_4axel_vagon <=  dist_1 <= max_koeff*l_base_under_4axel_vagon)\
            #                     and (min_koeff*l_base_4axel_wagon <=  dist_2 <= max_koeff*l_base_4axel_wagon)\
            #                     and (min_koeff*2*l_couple_4axel_wagon <=  dist_3 <= max_koeff*2*l_couple_4axel_wagon):
            #                         wagon_count = wagon_count + 1
            #                         for k in (0,2):
            #                             between_axels['wagon_count'].loc[base_4axel_indexes[i] + k] = wagon_count
            #                             between_axels['wagon_type'].loc[base_4axel_indexes[i] + k] = "4-osn wagon"
            #                             between_axels['AxlesAmount_InCar'].loc[base_4axel_indexes[i] + k] = 4
            #                         i = i+2
            #                         print ('Выявлен обычный 4-х осный вагон', i)

            #                     elif  (min_koeff*l_base_4axel_wagon <=  dist_2 <= max_koeff*l_base_4axel_wagon)\
            #                     and (min_koeff*(l_couple_4axel_wagon + l_couple_sochlenen_wagon)\
            #                          <=  dist_3 <= max_koeff*(l_couple_4axel_wagon + l_couple_sochlenen_wagon)):
            #                         wagon_count = wagon_count + 1
            #                         for k in (0,2):
            #                             between_axels['wagon_count'].loc[base_4axel_indexes[i] + k] = wagon_count
            #                             between_axels['wagon_type'].loc[base_4axel_indexes[i] + k] = "4-osn wagon"
            #                             between_axels['AxlesAmount_InCar'].loc[base_4axel_indexes[i] + k] = 4
            #                         i = i+2
            #                         print ('Выявлен обычный 4-х осный вагон', i)

            #                     elif (dist_1 >= min_koeff*2*l_couple_sochlenen_wagon)\
            #                     and (min_koeff*l_base_sochlenen_wagon <=  dist_2 <= max_koeff*l_base_sochlenen_wagon)\
            #                     and (dist_3 >=  min_koeff*2*l_couple_sochlenen_wagon)\
            #                     and (min_koeff*l_base_sochlenen_wagon <=  dist_4 <= max_koeff*l_base_sochlenen_wagon)\
            #                     and (min_koeff*2*l_couple_sochlenen_wagon <=  dist_5 <= max_koeff*2*l_couple_sochlenen_wagon):
            #                         wagon_count = wagon_count + 1
            #                         for k in (0,2,4):
            #                             between_axels['wagon_count'].loc[base_4axel_indexes[i] + k] = wagon_count
            #                             between_axels['wagon_type'].loc[base_4axel_indexes[i] + k] = "sochlenen 6-osn wagon"
            #                             between_axels['AxlesAmount_InCar'].loc[base_4axel_indexes[i] + k] = 6
            #                         i = i+3
            #                         print ('Выявлен сочлененный вагон', i)

            #                     elif (dist_1 >= min_koeff*(l_couple_4axel_wagon + l_couple_sochlenen_wagon))\
            #                     and (min_koeff*l_base_sochlenen_wagon <=  dist_2 <= max_koeff*l_base_sochlenen_wagon)\
            #                     and (dist_3 >=  min_koeff*(l_couple_4axel_wagon + l_couple_sochlenen_wagon))\
            #                     and (min_koeff*l_base_sochlenen_wagon <=  dist_4 <= max_koeff*l_base_sochlenen_wagon)\
            #                     and (min_koeff*(l_couple_4axel_wagon + l_couple_sochlenen_wagon)\
            #                          <=  dist_5 <= max_koeff*(l_couple_4axel_wagon + l_couple_sochlenen_wagon)):
            #                         wagon_count = wagon_count + 1
            #                         for k in (0,2,4):
            #                             between_axels['wagon_count'].loc[base_4axel_indexes[i] + k] = wagon_count
            #                             between_axels['wagon_type'].loc[base_4axel_indexes[i] + k] = "sochlenen 6-osn wagon"
            #                             between_axels['AxlesAmount_InCar'].loc[base_4axel_indexes[i] + k] = 6
            #                         i = i+3
            #                         print ('Выявлен сочлененный вагон', i)

            #                     else:
            #                         print ('Сочлененный или 4-х осный вагоны не выявлены', i)
            #                         between_axels['wagon_type'].loc[base_4axel_indexes[i]] = "unknown wagon type"
            #                         i = i + 1

            #                     if len(base_4axel_indexes) - i == 2:
            #                         wagon_count = wagon_count + 1
            #                         for k in (0,2):
            #                             between_axels['wagon_count'].loc[base_4axel_indexes[i] + k] = wagon_count
            #                             between_axels['wagon_type'].loc[base_4axel_indexes[i] + k] = "4-osn wagon"
            #                             between_axels['AxlesAmount_InCar'].loc[base_4axel_indexes[i] + k] = 4
            #                         i = i+2
            #                         print ('Выявлен обычный 4-х осный вагон', i)

            #                     if len(base_4axel_indexes) - i == 3:
            #                         wagon_count = wagon_count + 1
            #                         for k in (0,2,4):
            #                             between_axels['wagon_count'].loc[base_4axel_indexes[i] + k] = wagon_count
            #                             between_axels['wagon_type'].loc[base_4axel_indexes[i] + k] = "sochlenen 6-osn wagon"
            #                             between_axels['AxlesAmount_InCar'].loc[base_4axel_indexes[i] + k] = 6
            #                         i = i+3
            #                         print ('Выявлен сочлененный вагон', i)

            # #            Удаление ненужных межосевых расстояний
            #                 between_axels =  between_axels[ between_axels['wagon_count'].notna()]
            #                 between_axels.reset_index()

            # #                 удаление ненужных строк с тележками
            #                 wagon_counts = pd.unique(between_axels['wagon_count'])
            #                 wagon_counts.sort()
            #                 for wagon_count in wagon_counts:
            #                     wagon_indexes = between_axels['wagon_count'].loc[(between_axels['side']== side)\
            #                                                                      & (between_axels['wagon_count'] == wagon_count)].index
            #                     print (wagon_indexes)
            #                     print("len(wagon_indexes) = ", len(wagon_indexes))
            #                     print (between_axels)
            #                     max_wagon_axle = between_axels['axel_end'].loc[wagon_indexes[len(wagon_indexes)-1]]
            #                     between_axels['axel_end'].loc[wagon_indexes[0]] = max_wagon_axle

            #                     for i in range (1,len(wagon_indexes)):
            #                         between_axels = between_axels.drop(between_axels[(between_axels.index == wagon_indexes[i])].index)

            between_axels.reset_index()

            distances = [l_dpk1_tvs1, l_dpk1_tvs2, l_dpk1_tvs3, l_dpk1_tvs4, l_dpk1_tvs5, l_dpk1_tvs6]
            channel_numbs = [0, 1, 2, 4, 5, 6]  # Надо автоматизировать процесс создания этой матрицы номеров каналов
            if speed.mean() > 0:
                for i in range(len(distances)):
                    axel = pd.DataFrame(columns=['speed', 'numb_axel', 'sensor', 'time', 'side'])
                    axel['speed'] = np.abs(speed)
                    axel['numb_axel'] = axel.index
                    axel['sensor'] = channel_numbs[i]
                    axel['side'] = side
                    axel['time'] = times_i + distances[i] / speed
                    axels = pd.concat([axels, axel], ignore_index=True)
            elif speed.mean() < 0:
                for i in reversed(range(len(distances))):
                    axel = pd.DataFrame(columns=['speed', 'numb_axel', 'sensor', 'time', 'side'])
                    axel['speed'] = np.abs(speed)
                    axel['numb_axel'] = axel.index
                    axel['sensor'] = channel_numbs[i]
                    axel['side'] = side
                    # distance = int(distances[i])
                    axel['time'] = times_ii + (l_dpk12 - distances[i]) / speed
                    axels = pd.concat([axels, axel], ignore_index=True)
            else:

                print(
                    "Сторона: " + side + ".\n Сбой режима прохода осей через ДПК. Дальнейшая обработка записи невозможна!!")
                return (False)
            #                 axels = pd.DataFrame(columns=['speed', 'numb_axel', 'sensor', 'time', 'side'])
        #                 return axels

        if flag1 == "True":
            csv_name = self.unpak_dir + "\\" + self.neva_name + "_" + self.date + self.time.replace(":",
                                                                                                    "-") + "_axle_time_under_tvs.csv"
            axels.to_csv(csv_name, index=False, sep=';')
            csv_name = self.unpak_dir + "\\" + self.neva_name + "_" + self.date + self.time.replace(":",
                                                                                                    "-") + "_distance_between_accels.csv"

            del between_axels['dist_between']
            between_axels.to_csv(csv_name, index=False, sep=';', encoding="utf-8")

        return (axels, between_axels)

    def clear_from_dpk_counts(self, signal_data_freim, file_to_csv='False'):
        signal_df = signal_data_freim
        dpk_config = self.dpk_config
        flag1 = file_to_csv

        for i in (dpk_config.keys()):
            ch_i = (dpk_config.get(i))[0]
            ch_ii = (dpk_config.get(i))[1]
            side = str(i)
            signal_df = signal_df.drop(signal_df[(signal_df.sensor == int(ch_i)) & (signal_df.side == side)].index)
            signal_df = signal_df.drop(signal_df[(signal_df.sensor == int(ch_ii)) & (signal_df.side == side)].index)
            signal_df.reset_index()

        if flag1 == "True":
            csv_name = self.unpak_dir + "\\" + self.neva_name + "_" + self.date + self.time.replace(":",
                                                                                                    "-") + "data_from_tvs_only.csv"
            signal_df.to_csv(csv_name, index=False, sep=';')

        return signal_df

    def filter_signals(self, clear_signal_data_freim, file_to_csv='False'):
        isxod = clear_signal_data_freim
        config_algoritm = self.config_algoritm
        flag1 = file_to_csv

        # "Задание массива заголовков имен ДУЦ"

        sensorlist = pd.unique(isxod['sensor'])
        sensorlist.sort()
        sidelist = pd.unique(isxod['side'])
        sidelist.sort()

        # Фильтрация сигналов

        # Создание копий исходных массивов с сигналами
        sigs_filter = isxod.copy()
        sigs_filter['value_LF'] = -1000
        sigs_filter['value_HF'] = -1000

        # Присвоение переменных для функций фильтрации

        lowcut_LF, highcut_LF = config_algoritm.get("LF")[0], config_algoritm.get("LF")[1]
        lowcut_HF, highcut_HF = config_algoritm.get("HF")[0], config_algoritm.get("HF")[1]
        order_LF, order_HF = config_algoritm.get("LF")[2], config_algoritm.get("HF")[2]
        slot_rate_LF, slot_rate_HF = config_algoritm.get("LF")[3], config_algoritm.get("HF")[3]
        ld = config_algoritm.get("LD")
        #         print (lowcut_LF, highcut_LF)
        #         print (lowcut_HF, highcut_HF)
        #         print (order_LF, order_LF)
        #         print (slot_rate_LF, slot_rate_LF)
        #         print (ld)
        for side in sidelist:
            for sensor in sensorlist:
                data = sigs_filter['value'].loc[(sigs_filter['sensor'] == sensor) & (sigs_filter['side'] == side)]
                sigs_filter['value_LF'].loc[
                    (sigs_filter['sensor'] == sensor) & (sigs_filter['side'] == side)] = butter_bandpass_filter(data,
                                                                                                                lowcut_LF,
                                                                                                                highcut_LF,
                                                                                                                slot_rate_LF,
                                                                                                                order_LF)
                sigs_filter['value_HF'].loc[
                    (sigs_filter['sensor'] == sensor) & (sigs_filter['side'] == side)] = butter_bandpass_filter(data,
                                                                                                                lowcut_HF,
                                                                                                                highcut_HF,
                                                                                                                slot_rate_HF,
                                                                                                                order_HF)

        if flag1 == "True":
            csv_name = self.unpak_dir + "\\" + self.neva_name + "_" + self.date + self.time.replace(":",
                                                                                                    "-") + "_data_from_tvs_filtered.csv"
            sigs_filter.to_csv(csv_name, index=False, sep=';')

        return (sigs_filter)

    def delete_unuse_data(filter_signal_data_freim, axels, side, sensor):
        sigs_filter = filter_signal_data_freim.copy()
        accels = axels
        side = side
        sensor = sensor

        # "Задание массива сенсоров и сторонности до удаления"

        accellist = pd.unique(accels['numb_axel'])
        accellist.sort()
        sensorlist = pd.unique(sigs_filter['sensor'])
        sensorlist.sort()
        sidelist = pd.unique(sigs_filter['side'])
        sidelist.sort()

        #         Удаление из массивов элементов, не участвующих в выборе (лишняя сторона и лишний сенсор)

        #             print (len(sigs_filter), len(accels))
        for list_side in sidelist:
            if list_side != side:
                sigs_filter = sigs_filter.drop(sigs_filter[(sigs_filter.side == list_side)].index)
                sigs_filter.reset_index()
                accels = accels.drop(accels[(accels.side == list_side)].index)
                accels.reset_index()

        for sensor_id in sensorlist:
            if sensor_id != sensor:
                sigs_filter = sigs_filter.drop(sigs_filter[(sigs_filter.sensor == sensor_id)].index)
                sigs_filter.reset_index()
                accels = accels.drop(accels[(accels.sensor == sensor_id)].index)
                accels.reset_index()

        #             print (len(sigs_filter), len(accels))
        return (sigs_filter, accels)

    def create_all_stat(self, filtered_signal_data_freim, axels, file_to_csv='False'):
        isxod = filtered_signal_data_freim.copy()
        axels = axels.copy()
        config_algoritm = self.config_algoritm
        config_criteria = self.config_criteria
        train_id = self.train_id
        flag1 = file_to_csv
        ld = config_algoritm.get("LD")

        # Перекодирование символов сторонности в цифровой формат
        isxod['side'].loc[(isxod['side'] == 'R')] = 0
        isxod['side'].loc[(isxod['side'] == 'L')] = 1
        axels['side'].loc[(axels['side'] == 'R')] = 0
        axels['side'].loc[(axels['side'] == 'L')] = 1

        # "Задание массива номеров осей, сенсоров, сторонности"

        accellist = pd.unique(axels['numb_axel'])
        accellist.sort()
        sensorlist = pd.unique(isxod['sensor'])
        sensorlist.sort()
        sidelist = pd.unique(isxod['side'])
        sidelist.sort()

        # "Вычисление массива времен начала и окончания  зоны действия сенсора для номеров осей"

        axels['time_start'] = axels['time'] - ((ld / axels['speed']))
        axels['time_end'] = axels['time'] + ((ld / axels['speed']))

        # "Разбор исходных сигналов по осям и сенсорам. Фильтрация. Вычисление стат.параметров"

        Stat = np.zeros((len(accellist) * len(sensorlist) * len(sidelist), 17))
        Counts = len(accellist) * len(sensorlist) * len(sidelist)
        count = 0
        sdvig_hf = 0.02
        j = 0
        for side in sidelist:
            for i in sensorlist:
                freims = Neva2.delete_unuse_data(isxod, axels, side, i)
                sigs_filter = freims[0]
                accels = freims[1]
                for k in accellist:
                    #                     print (side)
                    #                     print (i)
                    #                     print (k)

                    Start_Time = accels['time_start'].loc[
                        (accels['sensor'] == i) & (accels['side'] == side) & (accels['numb_axel'] == k)].values

                    End_Time = accels['time_end'].loc[
                        (accels['sensor'] == i) & (accels['side'] == side) & (accels['numb_axel'] == k)].values

                    Times = sigs_filter['time'].loc[(sigs_filter['sensor'] == i) & (sigs_filter['side'] == side) & (
                                sigs_filter['time'] >= Start_Time[0]) & (sigs_filter['time'] <= End_Time[0])].values
                    Times_hf = sigs_filter['time'].loc[(sigs_filter['sensor'] == i) & (sigs_filter['side'] == side) & (
                                sigs_filter['time'] >= Start_Time[0] + sdvig_hf) & (sigs_filter['time'] <= End_Time[
                        0] + sdvig_hf)].values

                    values = sigs_filter['value'].loc[(sigs_filter['sensor'] == i) & (sigs_filter['side'] == side) & (
                                sigs_filter['time'] >= Start_Time[0]) & (sigs_filter['time'] <= End_Time[0])].values
                    values_LF = sigs_filter['value_LF'].loc[
                        (sigs_filter['sensor'] == i) & (sigs_filter['side'] == side) & (
                                    sigs_filter['time'] >= Start_Time[0] + sdvig_hf) & (
                                    sigs_filter['time'] <= End_Time[0] + sdvig_hf)].values
                    values_HF = sigs_filter['value_HF'].loc[
                        (sigs_filter['sensor'] == i) & (sigs_filter['side'] == side) & (
                                    sigs_filter['time'] >= Start_Time[0]) & (sigs_filter['time'] <= End_Time[0])].values

                    #                     plt.title('Значения values и values_LF для (сторона, №ТВС, №оси ): ' +\
                    #                               str(side) + ',' + str(i) + ',' + str(k) )
                    #                     plt.ylabel('Значение , т')
                    #                     plt.xlabel('Время, сек')
                    #                     plt.plot(Times, values, label = 'values')
                    #                     plt.plot(Times_hf, values_LF, label = 'values_LF')
                    #                     plt.legend()
                    #                     plt.show ()

                    #                     Start_Time_Index = sigs_filter['value_LF'].loc[(sigs_filter['sensor']== i)& (sigs_filter['side']== side)\
                    #                                                 & (sigs_filter['time']>= Start_Time[0])].index[0]

                    #                     End_Time_Index = sigs_filter['value_LF'].loc[(sigs_filter['sensor']== i)& (sigs_filter['side']== side)\
                    #                                                 & (sigs_filter['time']>= End_Time[0])].index[0]

                    #                     Start_Time_Index_HF = sigs_filter['value_HF'].loc[(sigs_filter['sensor']== i)& (sigs_filter['side']== side)\
                    #                                                 & (sigs_filter['time']>= Start_Time[0])].index[0]

                    #                     End_Time_Index_HF = sigs_filter['value_HF'].loc[(sigs_filter['sensor']== i)& (sigs_filter['side']== side)\
                    #                                                 & (sigs_filter['time']>= End_Time[0])].index[0]

                    Speed = accels['speed'].loc[
                        (accels['numb_axel'] == k) & (accels['side'] == side) & (accels['time'] >= Start_Time[0]) & (
                                    accels['time'] <= End_Time[0])].mean()
                    #                     print ('Скорость движения, м/с - ', Speed)
                    #                     print ()
                    #                     plt.title('Значения для сенсора - ' + str(sensor) + ', ось - ' + str(k) + ', сторона - ' + side)
                    #                     plt.ylabel('Значение , т')
                    #                     plt.xlabel('Время, сек')
                    #                     plt.plot(Times, values_LF)
                    #                     plt.plot(Times, values_HF)
                    #                     plt.show()

                    Stat[j][0] = 0
                    Stat[j][1] = 0
                    Stat[j][2] = int(train_id)
                    Stat[j][3] = int(max(accellist))
                    Stat[j][4] = side
                    Stat[j][5] = int(k)
                    Stat[j][6] = int(i)
                    Stat[j][7] = int(Speed * 3.6)
                    Stat[j][8] = max(values_LF)
                    Stat[j][9] = min(values)
                    Stat[j][10] = max(values)
                    Stat[j][11] = np.var(values)
                    Stat[j][12] = min(values_HF)
                    Stat[j][13] = max(values_HF)
                    Stat[j][14] = np.var(values_HF)
                    Stat[j][15] = int(0)
                    Stat[j][16] = int(0)

                    #                     print (Start_Time_Index, End_Time_Index, Start_Time_Index, End_Time_Index )
                    #                     sigs_filter.drop(sigs_filter.index[[Start_Time_Index,End_Time_Index]])
                    #                     sigs_filter.reset_index()
                    j = j + 1
                    count = count + 1
                    if (int(100 * count / Counts)) < 100:
                        print('Обработка и фильтрация исходных сигналов. Обработано_' + str(
                            int(100 * count / Counts)) + '_% данных', end="\r")
                    if (int(100 * count / Counts)) == 100:
                        print('Обработка и фильтрация исходных сигналов. Обработано_' + str(
                            int(100 * count / Counts)) + '_% данных')

        # Формирование датафрейма с исходными данными

        statistica = pd.DataFrame(Stat)
        statistica.columns = ['date', 'time', 'trainid', 'WheelCount', 'side', 'Axle_InTrain', 'DUS_Numb', 'speedd',
                              'Pst,t', 'min_val', 'max_val', 'disp_val', 'HFmin_val', 'HFmax_val', 'HFdisp_val',
                              'Trevoga_model', 'Trevoga_porogi']

        # Перекодировка сторонности ДУЦ от битов к символам

        statistica['side'].loc[(statistica['side'] == 0)] = 'R'
        statistica['side'].loc[(statistica['side'] == 1)] = 'L'

        if flag1 == "True":
            csv_name = self.unpak_dir + "\\" + self.neva_name + "_" + self.date + self.time.replace(":",
                                                                                                    "-") + "_all_statistica_from_tvs.csv"
            statistica.to_csv(csv_name, index=False, sep=';')

        return (statistica)

    def find_defected_wheels(self, all_statistica, all_wagon_list, file_to_csv='False'):

        statistica = all_statistica.copy()
        wagon_list = all_wagon_list.copy()
        flag1 = file_to_csv
        accellist = pd.unique(all_statistica['Axle_InTrain'])
        accellist.sort()
        sensorlist = pd.unique(all_statistica['DUS_Numb'])
        sensorlist.sort()
        sidelist = pd.unique(all_statistica['side'])
        sidelist.sort()

        for side in sidelist:
            #             print ("Сторона для которой определяются коэффициенты - ", side)
            hf_max_base_speed = ((self.config_criteria.get(side))[0])[0]
            hf_max_speed_degre = ((self.config_criteria.get(side))[0])[1]
            hf_max_base_pst = ((self.config_criteria.get(side))[0])[2]
            hf_max_pst_degre = ((self.config_criteria.get(side))[0])[3]
            hf_max = ((self.config_criteria.get(side))[0])[4]
            hf_max_k0 = ((self.config_criteria.get(side))[0])[5]
            #             print("Значения base_speed, speed_degree, base_pst, pst_degree, limit, border for hf_max = ",\
            #                  hf_max_base_speed, hf_max_speed_degre, hf_max_base_pst, hf_max, hf_max_k0)

            hf_min_base_speed = ((self.config_criteria.get(side))[1])[0]
            hf_min_speed_degre = ((self.config_criteria.get(side))[1])[1]
            hf_min_base_pst = ((self.config_criteria.get(side))[1])[2]
            hf_min_pst_degre = ((self.config_criteria.get(side))[1])[3]
            hf_min = ((self.config_criteria.get(side))[1])[4]
            hf_min_k0 = ((self.config_criteria.get(side))[1])[5]
            #             print("Значения base_speed, speed_degree, base_pst, pst_degree, limit, border for hf_min =",\
            #                  hf_min_base_speed, hf_min_speed_degre, hf_min_base_pst, hf_min, hf_min_k0)

            hf_disp_base_speed = ((self.config_criteria.get(side))[2])[0]
            hf_disp_speed_degre = ((self.config_criteria.get(side))[2])[1]
            hf_disp_base_pst = ((self.config_criteria.get(side))[2])[2]
            hf_disp_pst_degre = ((self.config_criteria.get(side))[2])[3]
            hf_disp = ((self.config_criteria.get(side))[2])[4]
            hf_disp_k0 = ((self.config_criteria.get(side))[2])[5]
            #             print("Значения base_speed, speed_degree, base_pst, pst_degree, limit, border for hf_disp =  ",\
            #                  hf_disp_base_speed, hf_disp_speed_degre, hf_disp_base_pst, hf_disp, hf_disp_k0)

            for accel in accellist:

                index = statistica['max_val'].loc[
                    (statistica['side'] == side) & (statistica['Axle_InTrain'] == accel) & (
                                statistica['DUS_Numb'] == min(sensorlist))].index[0]

                max_val = statistica['max_val'].loc[
                    (statistica['side'] == side) & (statistica['Axle_InTrain'] == accel) & (
                                statistica['DUS_Numb'] >= min(sensorlist)) & (
                                statistica['DUS_Numb'] <= max(sensorlist))].max()

                statistica['max_val'].loc[index] = max_val

                min_val = statistica['min_val'].loc[
                    (statistica['side'] == side) & (statistica['Axle_InTrain'] == accel) & (
                                statistica['DUS_Numb'] >= int(min(sensorlist))) & (
                                statistica['DUS_Numb'] <= int(max(sensorlist)))].min()
                statistica['min_val'].loc[index] = min_val

                disp_val = statistica['disp_val'].loc[
                    (statistica['side'] == side) & (statistica['Axle_InTrain'] == accel) & (
                                statistica['DUS_Numb'] >= min(sensorlist)) & (
                                statistica['DUS_Numb'] <= max(sensorlist))].max()
                statistica['disp_val'].loc[index] = disp_val

                hf_max_val = statistica['HFmax_val'].loc[
                    (statistica['side'] == side) & (statistica['Axle_InTrain'] == accel) & (
                                statistica['DUS_Numb'] >= min(sensorlist)) & (
                                statistica['DUS_Numb'] <= max(sensorlist))].max()
                statistica['HFmax_val'].loc[index] = hf_max_val

                hf_min_val = statistica['HFmin_val'].loc[
                    (statistica['side'] == side) & (statistica['Axle_InTrain'] == accel) & (
                                statistica['DUS_Numb'] >= int(min(sensorlist))) & (
                                statistica['DUS_Numb'] <= int(max(sensorlist)))].min()

                statistica['HFmin_val'].loc[index] = hf_min_val

                hf_disp_val = statistica['HFdisp_val'].loc[
                    (statistica['side'] == side) & (statistica['Axle_InTrain'] == accel) & (
                                statistica['DUS_Numb'] >= min(sensorlist)) & (
                                statistica['DUS_Numb'] <= max(sensorlist))].max()
                statistica['HFdisp_val'].loc[index] = hf_disp_val

                p_st = statistica['Pst,t'].loc[(statistica['side'] == side) & (statistica['Axle_InTrain'] == accel) & (
                            statistica['DUS_Numb'] >= min(sensorlist)) & (
                                                           statistica['DUS_Numb'] <= max(sensorlist))].values
                #                 print ("Номер оси = ", accel)
                #                 print ("Нaчальный массив Р_ст = ", p_st)
                p_st = sorted(p_st)
                #                 print ("Сортированный массив Р_ст = ", p_st)
                p_st.pop(len(p_st) - 1)
                p_st.pop(0)
                #                 print ("Очищенный массив Р_ст = ", p_st)
                #                 print ("Сумма элементов очищенного массива Р_ст = ", sum(p_st))
                #                 print ("Длина очищенного массива Р_ст = ", len(p_st))
                p_st = sum(p_st) / len(p_st)
                #                 print ("Средее очищенного массива Р_ст = ", p_st)
                statistica['Pst,t'].loc[index] = p_st
                statistica['DUS_Numb'].loc[index] = 10

                speed = statistica['speedd'].loc[
                    (statistica['side'] == side) & (statistica['Axle_InTrain'] == accel) & (
                                statistica['DUS_Numb'] >= min(sensorlist)) & (
                                statistica['DUS_Numb'] <= max(sensorlist))].mean()

                #              "Вычисление поправочных коэффициентов к пороговым значениям"
                #               "пороговых значений, определение уровня тревоги"
                #  Ki - поправочный коэффициент Ki к каждому базовому пороговому значению вычисляется по формуле
                #  Ki = ((Vi/Vb)**(1/Kv))* ((Pi/Pb)**Kp) где
                #  Vi - скорость движения колеса через сенсор
                #  Vb - базовая скорость, определяемая при настройке
                #  Kv - степенной коэффициент к отношению скоростей
                #  Pi - вес колеса, прошедшего через сенсор
                #  Pb - базовое значение веса колеса, определяемая при настройке
                #  Kp - степенной коэффициент к отношению весов колес

                koeff_hf_max = ((speed / hf_max_base_speed) ** (1 / hf_max_speed_degre)) * (
                            (p_st / hf_max_base_pst) ** hf_max_pst_degre)

                koeff_hf_min = ((speed / hf_min_base_speed) ** (1 / hf_min_speed_degre)) * (
                            (p_st / hf_min_base_pst) ** hf_min_pst_degre)

                koeff_hf_disp = ((speed / hf_disp_base_speed) ** (1 / hf_disp_speed_degre)) * (
                            (p_st / hf_disp_base_pst) ** hf_disp_pst_degre)

                T2_hf_max = hf_max * koeff_hf_max
                T2_hf_min = hf_min * koeff_hf_min
                T2_hf_disp = hf_disp * koeff_hf_disp
                T0_hf_max = T2_hf_max * hf_max_k0
                T0_hf_min = T2_hf_min * hf_min_k0
                T0_hf_disp = T2_hf_disp * hf_disp_k0

                #                 print ("Коэффициенты для hf_max, hf_min, hf_disp, сторона, ось № -",\
                #                        koeff_hf_max, koeff_hf_min, koeff_hf_disp, side, accel)
                #                 print ("Пороги Т2 для hf_max, hf_min, hf_disp, сторона, ось № -",\
                #                        T2_hf_max, T2_hf_min, T2_hf_disp, side, accel)
                #                 print ("Пороги Т0 для hf_max, hf_min, hf_disp, сторона, ось № -",\
                #                        T0_hf_max, T0_hf_min, T0_hf_disp, side, accel)

                if hf_min_val >= T0_hf_min and hf_max_val <= T0_hf_max and hf_disp_val <= T0_hf_disp:
                    statistica['Trevoga_porogi'].loc[index] = 0
                #                     print ()
                #                     print (Side,i,k,'T0')

                elif hf_min_val <= T2_hf_min and hf_max_val >= T2_hf_max and hf_disp_val >= T2_hf_disp:
                    statistica['Trevoga_porogi'].loc[index] = 2
                #                     print ()
                #                     print (Side,i,k,'T2')
                else:
                    statistica['Trevoga_porogi'].loc[index] = 1
        #                     print ()
        #                     print (Side,i,k,'T1')

        #         Удаление неинформативных столбцов
        statistica = statistica.drop(statistica[(statistica.DUS_Numb != 10)].index)
        statistica.reset_index()
        statistica['DUS_Numb'].loc[(statistica['DUS_Numb'] == 10)] = 0
        del statistica['DUS_Numb']

        # Заполнение данных по номерам вагонов, числу осей в вагоне, номеру оси в вагоне
        statistica['Car_number'] = 0
        statistica['AxlesAmount_InCar'] = 0
        statistica['AxleNumber_InCar'] = 0
        statistica['wagon_type'] = "none"

        for side in sidelist:
            for accel in accellist:
                wagon_list_index = wagon_list['axel_start'].loc[
                    (wagon_list['side'] == side) & (wagon_list['axel_start'] <= accel) & (
                                wagon_list['axel_end'] >= accel)].index
                axel_start = wagon_list['axel_start'].loc[
                    (wagon_list['side'] == side) & (wagon_list['axel_start'] <= accel) & (
                                wagon_list['axel_end'] >= accel)].values
                axel_end = wagon_list['axel_end'].loc[
                    (wagon_list['side'] == side) & (wagon_list['axel_start'] <= accel) & (
                                wagon_list['axel_end'] >= accel)].values

        #                 print (side, accel, axel_start, axel_end,  wagon_list_index)
        #                 print (wagon_list_index != wagon_list_index)
        # #

        # добавление даты,поезда, времени

        statistica['date'] = (self.date_time.split(' '))[0]
        statistica['time'] = (self.date_time.split(' '))[1]
        statistica['trainid'] = self.train_id
        statistica['WheelCount'] = statistica['WheelCount'] + 1
        statistica['Axle_InTrain'] = statistica['Axle_InTrain'] + 1

        #         Переименование столбцов под базу данных
        statistica.columns = ["Date", "Time", "TrainID", "WheelCount", "Side", "Axle_InTrain", "Speedd", "Pst",
                              "Min_val", "Max_val", "Disp_val", "HFMin_val", "HFMax_val", "HFDisp_val",
                              "Trevoga_model_max", "Trevoga_porogi_max", "Car_number", "AxlesAmount_InCar",
                              "AxleNumber_InCar", "Wagon_type"]

        if flag1 == "True":
            csv_name = self.unpak_dir + "\\" + self.neva_name + "_" + self.date + "_" + self.time.replace(":",
                                                                                                          "-") + "_statistica_free_from_tvs.csv"
            statistica.to_csv(csv_name, index=False, sep=';')

        return (statistica)

        # In[4]: