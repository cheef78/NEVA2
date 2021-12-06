# This is a sample Python script.
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
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.integrate import quad
from array import array
from numpy.fft import rfft, rfftfreq
from scipy.signal import sosfilt, sosfreqz
import datetime as dt
import yaml
from NEVA2 import Neva2
from user_functions import *

def main():
    try:

        config_dict = read_config('Neva2_config.yaml')

        slots_konfig = {'R': config_dict['slots_konfig_R'], 'L': config_dict['slots_konfig_L']}

        dpk_config = {'R': config_dict['dpk_config_R'], 'L': config_dict['dpk_config_R']}

        algoritm_config = {'LF': config_dict['algoritm_config_LF'], 'HF': config_dict['algoritm_config_HF'],
                           'LD': config_dict['algoritm_config_LD']}

        criteria_config = {'R': config_dict['criteria_config_R'], 'L': config_dict['criteria_config_L']}

        database_config = config_dict['database_config']
        neva_name = config_dict['neva_name']
        data_dir = config_dict['data_dir']
        unpak_dir = config_dict['unpak_dir']
        start_date_time = '"' + str(config_dict['start_date_time']) + '"'
        start_freze_time = config_dict['start_freze_time']
        db_control_period = config_dict['db_control_period']

    except Exception as e:
        error_code = 5
        msges = ''
        statuses = ''
        msg = 'Ошибка чтения файла конфигурации - ' + str(e) + " Код ошибки - " + str(error_code)
        status = "Обработка ПРЕРВАНА!!! Следующая попытка запуска алгоритма через 5 минут !!"
        msges = msges + ' ' + msg
        statuses = statuses + ' ' + status

        print(msges)
        print(statuses)
        print()
        time.sleep(300)

    # In[6]:


    unpack_colums = ['Date', 'Time', 'TrainID', 'FileName', 'error_code', 'msgs', 'status', 's_d_max_val', 's_d_min_val',
                     's_d_disp_val', 's_d_max_hfl', 's_d_min_hf', 's_d_disp_hf', 'T0_koeff', 'max_val', 'min_val',
                     'disp_val',
                     'max_hf', 'min_hf', 'disp_hf', 'base_speed', 'T2_count', 'T1_count']

    trains_upd_columns = ['WheelCount', 'Speed', 'SpeedL', 'ErrorCode', 'asoupStatus', 'Speed_max', 'Speed_min']


    k = 1
    a=1
    print('Пауза ' + str(start_freze_time) + ' сек для поднятия баз данных и прочих интерфейсов')
    print()
    time.sleep(start_freze_time)
    while a==1:
        # Поиск поездов с необработанными записями
        #     database_config = ["localhost","root","Spider@19128", "controlsystembase_dus"]
        try:
            msges = ''
            statuses = ''
            config_dict = read_config('Neva2_config.yaml')
            database_config = config_dict['database_config']
            connect = connect_database(database_config[0], database_config[1], database_config[2], database_config[3])
            start_date_time = '"' + str(config_dict['start_date_time']) + '"'
            #     start_date_time = "'2021-12-01 19:00:00'"
            trains_descs = []
            if connect[0] == True:
                engine = connect[1]
                train_list = pd.DataFrame((engine.execute(
                    'SELECT ID, DateTime, FileName FROM trains WHERE DateTime >= ' + start_date_time + ' AND asoupStatus = 0 Order By Id Desc')).fetchall())
                engine.dispose()

            else:
                error_code = 7
                msg = 'Ошибка доступа или соединения с БД - ' + str(connect[2]) + " Код ошибки - " + str(error_code)
                status = "Обработка ПРЕРВАНА!!! Следующая попытка запуска алгоритма через 5 минут !!"
                msges = msges + ' ' + msg
                statuses = statuses + ' ' + status

                print(msges)
                print(statuses)
                print()
                time.sleep(300)

            for elm in range(len(train_list.index)):
                train_desc = []
                train_desc = [
                    train_list[0][elm],
                    str(train_list[1][elm]),
                    train_list[2][elm]
                ]
                trains_descs.append(train_desc)

        except Exception as e:
            msg = 'Ошибка чтения данных из БД - ' + str(e)
            status = "Обработка ПРЕРВАНА!!! Следующая попытка запуска алгоритма через 5 минут !!"
            msges = msges + ' ' + msg
            statuses = statuses + ' ' + status
            error_code = 6
            print(msges)
            print(statuses)
            print()
            time.sleep(300)

        if len(trains_descs) != 0:
            print("В базе данных обнаружено не обработанных составов, шт = ", len(trains_descs), end="\r")
            print()
            for train_desc in trains_descs:
                k = 1
                try:
                    neva_name = config_dict['neva_name']

                    config_dict = read_config('Neva2_config.yaml')

                    slots_konfig = {'R': config_dict['slots_konfig_R'], 'L': config_dict['slots_konfig_L']}

                    dpk_config = {'R': config_dict['dpk_config_R'], 'L': config_dict['dpk_config_R']}

                    algoritm_config = {'LF': config_dict['algoritm_config_LF'], 'HF': config_dict['algoritm_config_HF'],
                                       'LD': config_dict['algoritm_config_LD']}

                    criteria_config = {'R': config_dict['criteria_config_R'], 'L': config_dict['criteria_config_L']}

                    database_config = config_dict['database_config']

                    data_dir = config_dict['data_dir']
                    unpak_dir = config_dict['unpak_dir']
                    msges = ''
                    statuses = ''

                    asoup_status = 0
                    error_code = []
                    t1_count = 0
                    t2_count = 0
                    speed_mean = 0
                    speed_min = 0
                    speed_max = 0
                    axle_numb = 0
                    delta_speed_mean = 0

                    neva_konfig = [neva_name, data_dir, unpak_dir, slots_konfig, dpk_config, algoritm_config,
                                   criteria_config, database_config, train_desc]
                    neva_system = Neva2(neva_konfig)
                    #     print()
                    #     neva_system.info()
                    print(
                        "Обрабатывается файл прохода поезда со следующими параметрами ID, ДАТА ПРОХОДА_ВРЕМЯ ПРОХОДА, путь обработки:",
                        train_desc)
                    signal_data_freim = neva_system.unpack('False')
                    axels = neva_system.find_wheel_time_speed(signal_data_freim, 'False')
                    if axels != False:
                        r_axels = axels[0]['numb_axel'].loc[(axels[0]['side'] == "R")].max() + 1
                        l_axels = axels[0]['numb_axel'].loc[(axels[0]['side'] == "L")].max() + 1
                        axle_numb = (r_axels + l_axels) / 2
                    if axels != False and r_axels == l_axels:
                        #  ВЫчисление среднего значения разницы скоростей правой и левой сторон для контроля
                        left_speed_max = 3.6 * (abs((axels[0]['speed'].loc[(axels[0]['side'] == 'L')].max())))
                        right_speed_max = 3.6 * (abs((axels[0]['speed'].loc[(axels[0]['side'] == 'R')].max())))
                        left_speed_min = 3.6 * (abs((axels[0]['speed'].loc[(axels[0]['side'] == 'L')].min())))
                        right_speed_min = 3.6 * (abs((axels[0]['speed'].loc[(axels[0]['side'] == 'R')].min())))
                        left_speed_mean = 3.6 * (abs((axels[0]['speed'].loc[(axels[0]['side'] == 'L')].mean())))
                        right_speed_mean = 3.6 * (abs((axels[0]['speed'].loc[(axels[0]['side'] == 'R')].mean())))
                        speed_min = abs(left_speed_min + right_speed_min) / 2
                        speed_max = abs(left_speed_max + right_speed_max) / 2
                        speed_mean = abs(left_speed_mean + right_speed_mean) / 2
                        delta_speed_mean = abs(left_speed_mean - right_speed_mean)

                        print("Cредние значения скоростей колес по показаниями ДПК правой и левой сторон, км/ч = ",
                              right_speed_mean, left_speed_mean)
                        print(
                            "Cреднее значение разницы скоростей колес между показаниями ДПК правой и левой сторон, км/ч = ",
                            delta_speed_mean)
                        if delta_speed_mean < 4:
                            msg = "Данные скоростей для колес ВАЛИДНЫ!! Разность скоростей, км/ч = " + str(delta_speed_mean)
                            msges = msges + ' ' + msg
                            print(msges)
                            #  Дальнейшая обработка
                            clear_signal_data_freim = neva_system.clear_from_dpk_counts(signal_data_freim, 'False')
                            filtered_signal_data_freim = neva_system.filter_signals(clear_signal_data_freim, 'False')
                            all_statistica_data_freim = neva_system.create_all_stat(filtered_signal_data_freim, axels[0],
                                                                                    'False')
                            free_statistica_data_freim = neva_system.find_defected_wheels(all_statistica_data_freim,
                                                                                          axels[1], 'False')
                            #    Определение числа тревог Т2 и Т1

                            t1_count = (
                            free_statistica_data_freim.loc[(free_statistica_data_freim['Trevoga_porogi_max'] == 1)].count()[
                                0])
                            t2_count = (
                            free_statistica_data_freim.loc[(free_statistica_data_freim['Trevoga_porogi_max'] == 2)].count()[
                                0])

                            asoup_status = 1
                            error_code = 0
                            status = "Обработка файлов завершена ПОЛНОСТЬЮ!!"
                            statuses = statuses + ' ' + status

                        else:

                            msg = "Данные скоростей для колес возможно НЕ ВАЛИДНЫ!! Разность скоростей, км/ч = " + str(
                                delta_speed_mean)
                            print(msges)
                            msges = msges + ' ' + msg
                            axle_numb = axle_numb
                            error_code = 1
                            asoup_status = 1
                            status = "Обработка файлов завершена ПОЛНОСТЬЮ !!!"
                            statuses = statuses + ' ' + status

                    else:

                        msg = 'Ошибка обработки при распознавании сигналов ДПК или счета осей !!!'
                        msges = msges + ' ' + msg
                        status = "Обработка ПРЕРВАНА !!!"
                        statuses = statuses + ' ' + status
                        asoup_status = 1
                        error_code = 2

                except Exception as e:
                    msg = 'Неизвестная ошибка обработки - ' + str(e)
                    status = "Обработка ПРЕРВАНА!!!"
                    msges = msges + ' ' + msg
                    statuses = statuses + ' ' + status
                    asoup_status = 1
                    error_code = 3
                print(msges)
                print(statuses)
                print()

                #  Внесение результатов в базу
                try:
                    if asoup_status == 1:

                        connect = connect_database(neva_system.database_config[0], neva_system.database_config[1],
                                                   neva_system.database_config[2], neva_system.database_config[3])
                        if connect[0] == False:
                            print("Запись результатов в базу данных не выполнена!!. Проблемы соединения с базой данных!!")

                        elif connect[0] == True:
                            engine = connect[1]

                            #  Удаление возможного дублирования записей в таблице результатов
                            engine.execute('DELETE FROM t_val_impulses WHERE TrainID = ' + str(neva_system.train_id))
                            engine.execute('DELETE FROM t_unpack WHERE TrainID = ' + str(neva_system.train_id))
                            if error_code == 0:
                                # Добавление записей в БД результатов по оценке дефектов
                                free_statistica_data_freim.to_sql('t_val_impulses', con=engine, if_exists='append',
                                                                  index=False)

                            #               Добавление записи по обработке в т_анпак
                            data_to_unpack = [
                                [(neva_system.date_time.split(' '))[0], (neva_system.date_time.split(' '))[1],
                                 neva_system.train_id,
                                 neva_system.file_full_name, error_code, msges, statuses,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 t2_count, t1_count]]

                            pd_to_unpack = pd.DataFrame(data_to_unpack, columns=unpack_colums)
                            pd_to_unpack.to_sql('t_unpack', con=engine, if_exists='append', index=False)

                            #                 Одновление записей в траинс

                            trains_upd_numbs = [int(axle_numb), int(speed_mean), int(delta_speed_mean), error_code,
                                                asoup_status, int(speed_max), int(speed_min)]

                            for item in range(len(trains_upd_columns)):
                                engine.execute('UPDATE trains SET ' + str(trains_upd_columns[item]) + ' = ' + str(
                                    trains_upd_numbs[item]) + ' WHERE ID = ' + str(neva_system.train_id))
                            engine.dispose()
                            print("Запись результатов в базу данных выполнена!!. Обработка данных поезда №_" + str(
                                neva_system.train_id) + " Завершена успешно !!!")
                            print()

                except Exception as e:
                    msg = 'Неизвестная ошибка при внесении записей в базу данных - ' + str(e)
                    error_code = 4
                    print(msg)
                    print()

                    #     else:
        #         print ("Новых записей для составов в базе данных не обнаружено !!!")
        #         print()
        else:
            try:
                config_dict = read_config('Neva2_config.yaml')
                db_control_period = config_dict['db_control_period']
                print('Время ожидания поступления в базу новых составов  = ' + str(k * db_control_period / 60) + ' мин', end="\r")
                #     print()
                time.sleep(db_control_period)
                k += 1
            except Exception as e:
                msges = ''
                statuses = ''
                msg = 'Неизвестная ошибка чтения конфигурации - ' + str(e)
                status = "Обработка ПРЕРВАНА!!! Следующая попытка запуска алгоритма через 5 минут !!"
                msges = msges + ' ' + msg
                statuses = statuses + ' ' + status
                error_code = 5
                print(msges)
                print(statuses)
                print()
                time.sleep(300)












    # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
