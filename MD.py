#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import json
import sys
import PyQt5
import os
import pjlsa
import pyjapc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dateutil import tz
from PyQt5.QtWidgets import QMainWindow, QApplication, QCompleter, QFileDialog
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from ui_interface import Ui_MainWindow

from matplotlib.backends.qt_compat import is_pyqt5
if is_pyqt5():
    print("Using Qt5")
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    print("Using Qt4")
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

path_distance = lambda r,c: np.sum([np.linalg.norm(c[r[p]]-c[r[p-1]]) for p in range(len(r))])
two_opt_swap = lambda r,i,k: np.concatenate((r[0:i],r[k:-len(r)+i-1:-1],r[k+1:len(r)]))

def two_opt(cities,improvement_threshold, progress_bar=None):
    route = np.arange(cities.shape[0])
    improvement_factor = 1
    best_distance = path_distance(route,cities)
    if progress_bar != None:
            progress_bar.setValue(0)
    while improvement_factor > improvement_threshold:
        distance_to_beat = best_distance
        for swap_first in range(1,len(route)-2):
            for swap_last in range(swap_first+1,len(route)):
                new_route = two_opt_swap(route,swap_first,swap_last)
                new_distance = path_distance(new_route,cities)
                if new_distance < best_distance:
                    route = new_route
                    best_distance = new_distance #
        improvement_factor = 1 - best_distance/distance_to_beat
        if progress_bar != None:
            percentage = 100*(best_distance/distance_to_beat)
            progress_bar.setValue(percentage)
    if progress_bar != None:
        progress_bar.setValue(100)
    return route


def opti_1D(points):
    points = np.array(points)
    sort_ind = np.argsort(points)
    
    start_pos = np.argwhere(sort_ind==0).ravel().tolist()[0]
    
    sort_ind = np.append(sort_ind[0:start_pos+1][::-1], sort_ind[start_pos+1:])
    sort_ind_temp = np.append(sort_ind[0:start_pos+1:2], sort_ind[1:start_pos+1:2][::-1])
    sort_ind = np.append(sort_ind_temp, sort_ind[start_pos+1:])
    
    start_pos = np.argwhere(sort_ind==len(points)-1).ravel().tolist()[-1]
    
    sort_ind = np.append(sort_ind[0:start_pos], sort_ind[start_pos:len(sort_ind)][::-1])
    if start_pos % 2:    
        sort_ind_temp = np.append(sort_ind[start_pos+1::2][::-1], sort_ind[start_pos::2])
    else:
        sort_ind_temp = np.append(sort_ind[start_pos::2][::-1], sort_ind[start_pos+1::2])
    sort_ind = np.append(sort_ind[:start_pos], sort_ind_temp)
    return points[sort_ind]


def wait_time_later(reset, wait):
    return reset + wait < time.time()

def compare_dict(dict1, dict2):
    equal = True
    if dict1.keys() != dict2.keys():
        equal = False
    else:
        for k, v in dict1.items():
            if np.any(dict1[k] != dict2[k]):
                equal = False
    return equal

class Worker(QThread):

    signal = pyqtSignal(int)
    running = pyqtSignal(bool)
    get_lifetime = pyqtSignal(bool)

    def __init__(self):
        super(Worker, self).__init__()
        self.threadactive = True
                
    def run(self):
        if not self.threadactive:
            self.threadactive = True

        self.running.emit(self.threadactive)
        # to not skip first trim
        # get lifetime
        self.get_lifetime.emit(False)
        t1 = time.time()
        # non blocking wait
        while wait_time_later(t1, self.waittime) == False and self.threadactive == True:
            time.sleep(0.1)
        # write lifetime after wait
        if self.threadactive:
            self.get_lifetime.emit(True)
        
        while self.sequence.current < self.sequence.length()-1  and self.threadactive == True:
            # get lifetime before trim
            self.get_lifetime.emit(False)
            self.sequence.next_trim()
            self.signal.emit(self.sequence.current)
            t1 = time.time()
            # non blocking wait
            while wait_time_later(t1, self.waittime) == False and self.threadactive == True:
                time.sleep(0.1)
            # get lifetime after trim
            if self.threadactive:
                self.get_lifetime.emit(True)

        print("finished...")
        self.threadactive = False
        self.running.emit(self.threadactive)

    def stop(self):
        print("stopping")
        self.threadactive = False
        self.running.emit(self.threadactive)
        self.wait()

# this class is unused 
class Pyjapc_fetcher(QThread):

    newDataReceived = pyqtSignal('PyQt_PyObject')

    def __init__(self, japc, fesaproperties=None, time_interval=20):
        super(Pyjapc_fetcher, self).__init__()
        self.pyjapc = japc
        self.properties = fesaproperties
        self.time_interval = time_interval
        self.x = {}
        self.y = {}
        for p in fesaproperties:
            self.x[p] = []
            self.y[p] = []
            
    def __del__(self):
        self.close()

    def run(self):
        print("run")
        self.pyjapc.enableInThisThread()
        for p in self.properties:
            self.pyjapc.subscribeParam(p, self.onValueReceived , getHeader=True, unixtime=False)
            self.pyjapc.startSubscriptions()

    def onValueReceived(self, japc_cmd, value, headerInfo):
        print("value received")
        # store the values and output the x and y array
        currentDatetimeUTC = headerInfo['acqStamp']
        currentDatetime = currentDatetimeUTC.astimezone(tz.tzlocal())
        currentTimestamp = time.mktime(currentDatetime.timetuple())

        # reset the lifetime arrays
        if self.x[japc_cmd] and currentTimestamp > self.x[japc_cmd][0] + self.time_interval:
            self.x[japc_cmd] = []
            self.y[japc_cmd] = []

        self.y[japc_cmd].append(value)
        self.x[japc_cmd].append(currentTimestamp)

        self.newDataReceived.emit([self.x, self.y])
        # this should connect directly to plotting function

    def close(self):
        self.pyjapc.stopSubscriptions()
        self.pyjapc.clearSubscriptions()
        print("Closing Pyjapc fetcher")


class AppWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(AppWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.figure = plt.figure()
        self.ui.canvas = FigureCanvas(self.ui.figure)
        self.ui.verticalLayout_2.insertWidget(0,self.ui.canvas)
        self.ui.toolbar = NavigationToolbar(self.ui.canvas, self)
        self.ui.toolbar.setParent(self.ui.canvas)

        self.ui.figure_2 = plt.figure()
        self.ui.canvas_2 = FigureCanvas(self.ui.figure_2)
        self.ui.verticalLayout_5.insertWidget(0,self.ui.canvas_2)
        self.ui.toolbar_2 = NavigationToolbar(self.ui.canvas_2, self)
        self.ui.toolbar_2.setParent(self.ui.canvas_2)

        # live feed of lifetimes reset every trim set ?
        self.ui.figure_lt = plt.figure()
        self.ui.canvas_lt = FigureCanvas(self.ui.figure_lt)
        self.ui.layout_lifetime.insertWidget(0,self.ui.canvas_lt)
        self.ui.toolbar_lt = NavigationToolbar(self.ui.canvas_lt, self)
        self.ui.toolbar_lt.setParent(self.ui.canvas_lt)

        self.show()

        # Beam process completer
        self.lsa = pjlsa.LSAClient(server='gpn') # change to server="lhc" for setting trims
        self.bp_list = self.lsa.findBeamProcesses()
        self.completer = QCompleter(self.bp_list, self.ui.line_bp)
        self.ui.line_bp.setCompleter(self.completer)

        self.ui.lineEdit_search.setText("LHCBEAM1/TRIM")
        self.ui.line_n_points.setText(str(100))
        self.ui.line_waittime.setText(str(20))
        self.ui.line_outfile.setText(os.getcwd()+"/log")
        resident_bp = self.lsa.getResidentBeamProcess("POWERCONVERTERS")
        self.ui.line_bp.setText("SQUEEZE-6.5TeV-ATS-1m-30cm-2018_V1@638_[END]")
        self.ui.line_desc.setText("MD4510")
        self.ui.progressBar.hide()

        self.worker = Worker()
        self.worker.signal.connect(self.step_logic)
        self.worker.running.connect(self.enable_disable)
        self.worker.get_lifetime.connect(self.lifetime_logic)

        self.ui.pb_search.clicked.connect(self.search_click)
        self.ui.pb_search_add.clicked.connect(self.add_click)
        self.ui.pb_generate.clicked.connect(self.draw_trims)
        self.ui.pb_ref.clicked.connect(self.reference_trim)
        self.ui.pb_clear.clicked.connect(self.pb_clear_logic)
        self.ui.pb_tunes.clicked.connect(self.add_tunes)
        self.ui.pb_chrom.clicked.connect(self.add_chrom)
        self.ui.pb_oct.clicked.connect(self.add_oct)
        self.ui.pb_optimize.clicked.connect(self.optimize_draw)
        self.ui.pb_save.clicked.connect(self.saveFileDialog)
        self.ui.pb_load.clicked.connect(self.openFileNameDialog)
        self.ui.comboBox.currentTextChanged.connect(self.combo_plot)
        self.ui.comboBox_2.currentTextChanged.connect(self.combo_plot)

        self.ui.pb_go.clicked.connect(self.start_sequence)
        self.ui.pb_stop.clicked.connect(self.stop_sequence) # this would also connect to pyjapc_fetcher.stop
        self.ui.pb_revert.clicked.connect(self.back_sequence)
        self.ui.pb_forward.clicked.connect(self.forward_sequence)
        self.ui.pb_initial.clicked.connect(self.initial_sequence)
        self.ui.pb_clear_lt.clicked.connect(self.clear_lt)

        self.ui.tabWidget.currentChanged.connect(self.populate_combo)
        self._column_headers = ["parameter", "min range", "max range"]
        self.ui.tableWidget.setHorizontalHeaderLabels(self._column_headers)
        self.ui.tableWidget.horizontalHeader().setSectionResizeMode(0, PyQt5.QtWidgets.QHeaderView.Stretch)
        self.ui.tableWidget.horizontalHeader().setSectionResizeMode(1, PyQt5.QtWidgets.QHeaderView.ResizeToContents)
        self.ui.tableWidget.horizontalHeader().setSectionResizeMode(2, PyQt5.QtWidgets.QHeaderView.ResizeToContents)

        self.trims = {}
        self.extra_item = {"Beam 1 Tune diagram": ["LHCBEAM1/QH_TRIM", "LHCBEAM1/QV_TRIM"],
                           "Beam 2 Tune diagram": ["LHCBEAM2/QH_TRIM", "LHCBEAM2/QV_TRIM"]
                           }
        self.lifetime_history = []

       # rbac login for lifetime fetching
        self.japc = pyjapc.PyJapc( noSet = True )
        self.japc.setSelector(None)
        self.japc.rbacLogin(username='lcoyle') #set username here

    @property
    def selected_params(self):
        n_rows = self.ui.tableWidget.rowCount()
        selected_params = []
        for i in range(n_rows):
            selected_params.append(self.ui.tableWidget.item(i,0).text())
#        print("selected_params: {}".format(selected_params))
        return selected_params

    @property
    def selected_table(self):
        number_of_rows = self.ui.tableWidget.rowCount()
        number_of_columns = self.ui.tableWidget.columnCount()

        tmp_df = pd.DataFrame(
                    columns=[i.replace(' ', '_') for i in self._column_headers], # Fill columns
                    index=range(number_of_rows) # Fill rows
                    )
        for i in range(number_of_rows):
            for j in range(number_of_columns):
                tmp_df.ix[i, j] = self.ui.tableWidget.item(i, j).text()
        return tmp_df

    def param_min_max(self, param):
        table = self.selected_table
        row = table.loc[table["parameter"]==param]
        ref = self.trims[param][0]
        return (float(row["min_range"])+ref, float(row["max_range"])+ref)

    def search_click(self):
        srch_str = self.ui.lineEdit_search.text()
        self.ui.list_search.clear()
        srch_str = srch_str.split('/')
        device = srch_str[0]
        if len(srch_str) > 1:
            regx = srch_str[-1]
        else:
            regx = ''
        parameter_srch_out = self.lsa.findParameterNames(deviceName=device, regexp=regx)
        self.ui.list_search.addItem(device)
        for i in parameter_srch_out:
            self.ui.list_search.addItem(i)

    def add_tunes(self):
        for i in ["LHCBEAM1/QH_TRIM", "LHCBEAM1/QV_TRIM",
                  "LHCBEAM2/QH_TRIM", "LHCBEAM2/QV_TRIM"]:
            if i not in self.selected_params:
                self._add_full_row(i, -0.01, 0.01)

    def add_chrom(self):
        for i in ["LHCBEAM1/QPH_TRIM","LHCBEAM1/QPV_TRIM","LHCBEAM2/QPH_TRIM","LHCBEAM2/QPV_TRIM"]:
            if i not in self.selected_params:
                self._add_full_row(i, 0, 0)

    def add_oct(self):
        for i in ["LHCBEAM1/LANDAU_DAMPING_ROF", "LHCBEAM1/LANDAU_DAMPING_ROD", 
                  "LHCBEAM2/LANDAU_DAMPING_ROF", "LHCBEAM2/LANDAU_DAMPING_ROD"]:
            if i not in self.selected_params:
                self._add_full_row(i, 0, 0)

    def add_click(self):
        selected = self.ui.list_search.selectedItems()
        for p in selected:
            self._add_full_row(p.text(), 0, 0)

    def pb_clear_logic(self):
         selected = set([index.row() for index in self.ui.tableWidget.selectedIndexes()])
         if len(selected) == 0:
             self._remove_all_rows()
         else:
             self._remove_selected_rows(selected)

    def populate_combo(self):

        if self.ui.tabWidget.currentIndex() == 1:
            if set(self.trims.keys()) != set(self.selected_params):
                missing = list(set(self.selected_params) - set(self.trims.keys()))
#                print("Missing: {}".format(missing))
                self.draw_trims(params=missing)
                extra = list(set(self.trims.keys()) - set(self.selected_params))
#                print("Extra: {}".format(extra))
                for p in extra:
                    self.trims.pop(p)
#                print(self.trims.keys())

            if len(self.selected_params) == 0:
                self.ui.comboBox.clear()
                self.ui.comboBox.addItem("Please select parameters")
                self.ui.figure.clf()
                self.ui.canvas.draw()
            else:
                self.ui.comboBox.clear()
                self.ui.comboBox.addItems(self.selected_params)
                for k,v in self.extra_item.items():
                    if set(v).issubset(set(self.selected_params)):
                        self.ui.comboBox.addItem(k)

        if self.ui.tabWidget.currentIndex() == 2:
            if set(self.trims.keys()) != set(self.selected_params):
                extra = list(set(self.trims.keys()) - set(self.selected_params))
                for p in extra:
                    self.trims.pop(p)
#                print(self.trims.keys())

            if len(self.selected_params) == 0:
                self.ui.comboBox_2.clear()
                self.ui.comboBox_2.addItem("Please select parameters")
                self.ui.figure_2.clf()
                self.ui.canvas_2.draw()
            else:
                self.ui.comboBox_2.clear()
                for k, _ in self.trims.items():
                    self.ui.comboBox_2.addItem(k)
                for k,v in self.extra_item.items():
                    if set(v).issubset(set(self.trims.keys())):
                        self.ui.comboBox_2.addItem(k)

    def _remove_all_rows(self):
        n_rows = self.ui.tableWidget.rowCount()
        for i in reversed(range(n_rows)):
            self._remove_row(i)

    def _remove_selected_rows(self, selected):
        selected = sorted(selected)
        for row in reversed(selected):
            self._remove_row(row)

    def _remove_row(self, indice):
        self.ui.tableWidget.removeRow(indice)

    def _add_full_row(self, param, pmin, pmax):
        numRows = self.ui.tableWidget.rowCount()
        self.ui.tableWidget.insertRow(numRows)
        # Add text to the row
        self.ui.tableWidget.setItem(numRows, 0, QtWidgets.QTableWidgetItem(param))
        self.ui.tableWidget.setItem(numRows, 1, QtWidgets.QTableWidgetItem(str(pmin)))
        self.ui.tableWidget.setItem(numRows, 2, QtWidgets.QTableWidgetItem(str(pmax)))

    def draw_trims(self, params=False):
        if params == False:
            params = self.selected_params
            #print("selected_params: {}".format(self.selected_params))

        table = self.selected_table
        table = table.loc[table["parameter"].isin(params)]
        for i, p in table.iterrows():
            ranges = [float(p["min_range"]), float(p["max_range"])]
            self.trims[p["parameter"]] = [self.draw_random(ranges) for i in range(int(self.ui.line_n_points.text()))]
            self.trims[p["parameter"]].insert(0, 0)
            self.trims[p["parameter"]].append(0)
#        print(self.trims.keys())

        self.reset_worker()
        self.combo_plot()

    def reference_trim(self):
        reference_trims = self.fetch_latest_trims()
        print("Latest trims: {}".format(reference_trims))
        for k,v in self.trims.items():
            self.trims[k] = [i + reference_trims[k] - v[0] for i in v]
        self.reset_worker()
        self.combo_plot()

    def draw_random(self, rand_range):
        rand = np.random.uniform(low=min(rand_range), high=max(rand_range))
        return rand

    def reset_worker(self):
        if not hasattr(self.worker, 'sequence') or compare_dict(self.worker.sequence.values, self.trims) == False:
            self.worker.sequence = TrimSequencer(self.trims.copy())
#            print("Reseting Worker")
        if not hasattr(self.worker, 'waittime') or self.worker.waittime != float(self.ui.line_waittime.text()):
            self.worker.waittime = float(self.ui.line_waittime.text())

    def combo_plot(self):
        if self.ui.tabWidget.currentIndex() == 1:
            combobox = self.ui.comboBox
            figure = self.ui.figure
            canvas = self.ui.canvas
        elif self.ui.tabWidget.currentIndex() == 2:
            combobox = self.ui.comboBox_2
            figure = self.ui.figure_2
            canvas = self.ui.canvas_2
        else:
            return

        sel_param = combobox.currentText()
        if sel_param in self.trims.keys():
            figure.clf()
            ax = figure.gca()
            ax.plot(self.trims[sel_param], '*-')
            if hasattr(self.worker, 'sequence'):
                ax.plot(self.worker.sequence.current,
                        self.worker.sequence.return_value()[sel_param], '.',
                        ms=14, c='red')
            ax.set_xlabel("Point number")
            ax.set_ylabel("")
            canvas.draw()

        if sel_param in list(self.extra_item.keys()):
            figure.clf()
            ax = figure.gca()
            x = self.trims[self.extra_item[sel_param][0]]
            y = self.trims[self.extra_item[sel_param][1]]
            ax.plot(x, y, lw=1, marker=".")
            if hasattr(self.worker, 'sequence'):
                ax.plot(x[self.worker.sequence.current],
                        y[self.worker.sequence.current], '.', ms=14, c='red')
            ax.set_xlabel("Qx")
            ax.set_ylabel("Qy")
            canvas.draw()

    def lifetime_plot(self):
        
        self.lifetime_history.append(self.lifetime)
        self.ui.figure_lt.clf()
#        print(self.worker.sequence.current)
#        print(self.trims["LHCBEAM1/QH_TRIM"])
#        print([i['lifetimeB1'] for i in self.lifetime_history])
#        print(self.lifetime)
        if set(["LHCBEAM1/QH_TRIM", "LHCBEAM1/QV_TRIM"]).issubset(set(self.trims.keys())):
            ax1 = self.ui.figure_lt.add_subplot(2,1,1)
            cbar = ax1.scatter(self.trims["LHCBEAM1/QH_TRIM"][self._start_index:self.worker.sequence.current],
                        self.trims["LHCBEAM1/QV_TRIM"][self._start_index:self.worker.sequence.current],
                        c=[i['lifetimeB1'] for i in self.lifetime_history], s=14,
                        cmap='Spectral')
            
            ax1.set_xlim(self.param_min_max("LHCBEAM1/QH_TRIM"))
            ax1.set_ylim(self.param_min_max("LHCBEAM1/QV_TRIM"))
            ax1.set_title("Beam 1 tune/lifetime feed")
            ax1.set_xlabel("Qx")
            ax1.set_ylabel("Qy")
            self.ui.figure.colorbar(cbar, ax=ax1)
        if set(["LHCBEAM2/QH_TRIM", "LHCBEAM2/QV_TRIM"]).issubset(set(self.trims.keys())):
            ax2 = self.ui.figure_lt.add_subplot(2,1,2)
            cbar = ax2.scatter(self.trims["LHCBEAM2/QH_TRIM"][self._start_index:self.worker.sequence.current],
                        self.trims["LHCBEAM2/QV_TRIM"][self._start_index:self.worker.sequence.current],
                        c=[i['lifetimeB2'] for i in self.lifetime_history], s=14, 
                        cmap='Spectral')
            
            ax2.set_xlim(self.param_min_max("LHCBEAM2/QH_TRIM"))
            ax2.set_ylim(self.param_min_max("LHCBEAM2/QV_TRIM"))
            ax2.set_title("Beam 2 tune/lifetime feed")
            ax2.set_xlabel("Qx")
            ax2.set_ylabel("Qy")
            self.ui.figure.colorbar(cbar, ax=ax2)
        
        self.ui.canvas_lt.draw()

    def clear_lt(self):
        self.lifetime_history = []
        self.ui.figure_lt.clf()
        self.ui.canvas_lt.draw()

    def optimize_draw(self):
        sel_param = self.ui.comboBox.currentText()
        self.ui.progressBar.show()
        if sel_param in self.extra_item.keys():
            tunes = list(zip(self.trims[self.extra_item[sel_param][0]][:-1],
                                      self.trims[self.extra_item[sel_param][1]][:-1]))
            route = two_opt(np.array(tunes), 0.001, progress_bar=self.ui.progressBar)

            new_cities_order = np.concatenate((np.array([tunes[route[i]] for i in range(len(route))]),
                                           np.array([tunes[0]])))

            self.trims[self.extra_item[sel_param][0]] = new_cities_order[:,0].tolist()
            self.trims[self.extra_item[sel_param][1]] = new_cities_order[:,1].tolist()

        else:
#            route = two_opt(np.array(self.trims[sel_param][:-1]), 0.001, progress_bar=self.ui.progressBar)
#            new_cities_order = np.concatenate((np.array([self.trims[sel_param][route[i]] for i in range(len(route))]),
#                                           np.array([self.trims[sel_param][0]])))
            
            self.trims[sel_param] = opti_1D(self.trims[sel_param])

        self.ui.progressBar.hide()
        self.reset_worker()
        self.combo_plot()

    def enable_disable(self, running):
        self.ui.pb_go.setEnabled(not running)
        self.ui.pb_revert.setEnabled(not running)
        self.ui.line_outfile.setEnabled(not running)
        self.ui.line_waittime.setEnabled(not running)
        self.ui.line_bp.setEnabled(not running)
        self.ui.pb_forward.setEnabled(not running)
        self.ui.pb_initial.setEnabled(not running)
        self.ui.check_simulate.setEnabled(not running)
        self.ui.pb_save.setEnabled(not running)
        self.ui.pb_load.setEnabled(not running)
        self.ui.pb_generate.setEnabled(not running)
        self.ui.pb_optimize.setEnabled(not running)
        self.ui.pb_ref.setEnabled(not running)
        self.ui.line_desc.setEnabled(not running)


    def start_sequence(self):
        self.lifetime = self.fetch_lifetime()
        self.write_log("STARTED")
        self.reset_worker()
        self.worker.start()
        self._start_index = self.worker.sequence.current
        #self.lifetime_fetcher.start()	

    def stop_sequence(self):
        self.write_log("STOPPED")
        self.worker.stop()
        #self.lifetime_fetcher.start()	

    def back_sequence(self):
        if self.worker.sequence.current > 0:
            self.write_log("REVERT")
            self.worker.sequence.current -= 1
            self.step_logic(self.worker.sequence.current)

    def forward_sequence(self):
        if self.worker.sequence.current < self.worker.sequence.length()-1:
            self.write_log("FORWARD")
            self.worker.sequence.current += 1
            self.step_logic(self.worker.sequence.current)

    def initial_sequence(self):
        self.write_log("INITIAL")
        self.worker.sequence.current = 0
        self.step_logic(self.worker.sequence.current)

    def lifetime_logic(self, write):
        self.prev_lifetime = self.lifetime
        self.lifetime = self.fetch_lifetime()
        if write:
            self.lifetime_plot()
            string = "{}: wait_time: {}, prev_lifetime: {}, lifetime: {}".format(self.worker.sequence.current,
                      self.worker.waittime, self.prev_lifetime, self.lifetime)
            for k,v in self.worker.sequence.return_value().items():
                string = string + ' {}: {},'.format(k,v)
            print(string)
            self.write_log(string)

    def step_logic(self, step):
        # plot dot on sequence
        self.combo_plot()
        #apply trim
        self.apply_trim(self.ui.line_bp.text(), desc='MD4510')


    def fetch_lifetime(self):
        fetch = self.japc.getParam(["LHC.BLM.LIFETIME/Acquisition#lifetimeB1",
                                    "LHC.BLM.LIFETIME/Acquisition#lifetimeB2"])
        lifetimes = {}
        lifetimes["lifetimeB1"] = fetch[0]
        lifetimes["lifetimeB2"] = fetch[1]
        # for testing
#        lifetimes = {'lifetimeB1':time.time(),
#                     'lifetimeB2':time.time()}
        return lifetimes

    def fetch_latest_trims(self, params=None):
        if params == None:
            params = self.selected_params
 #       t0 = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(time.time()-3600)) # Is this the best way of getting the latest trims ? 
 #       t1 = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(time.time()))
        trims = {}
        for param in params:
            trims[param] = self.lsa.getLastTrim(beamprocess=self.ui.line_bp.text(), parameter=param).data
#        print(trims)
        #for k, v in trims.items():
        #    trims[k] = trims[k][1][-1]
        return trims

    def apply_trim(self, bp, desc='testing',  relative=False):
        trims = self.worker.sequence.return_value()
        param_list = []
        value_list = []
        for k, v in trims.items():
            param_list.append(k)
            value_list.append(v)
        param_list = self.lsa._buildParameterList(param_list)
        self.lsa.setTrims(bp, param_list, value_list, desc, relative=relative,
                          simulate=self.ui.check_simulate.isChecked())

    def write_log(self, string):
        with open(self.ui.line_outfile.text(), 'a+') as outfile:
            outfile.write(string+'\n')
            #outfile.flush()
    
    def openFileNameDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Load sequence", "","Json files (*.json);;All Files (*)", options=options)
        if fileName:
            print("Loading {}".format(fileName))
            self.load_sequence(fileName)
    
    def saveFileDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"Save sequence","","Json files (*.json);;All Files (*)", options=options)
        if fileName:
            print("Saving to {}".format(fileName))
            self.write_sequence(fileName)
            
    def write_sequence(self, file_path):
        with open(file_path, 'w') as outfile:
            json.dump(self.trims, outfile)
    
    def load_sequence(self, file_path):
        with open(file_path, 'r') as outfile:
            self.trims = json.load(outfile)
        self.reset_worker()
        self.combo_plot()
        self.populate_combo()
            
            
class TrimSequencer(object):

    def __init__(self, values):
        self.values = values
        self.current = 0

    def length(self):
        if self.values.keys() == []:
            return 0
        else:
            return len(self.values[list(self.values.keys())[0]])

    def next_trim(self):
        self.current += 1
        self.return_value()

    def prev_trim(self):
        self.current -= 1
        self.return_value()

    def current_trim(self):
        self.return_value()

    def reset(self):
        self.current = 0
        self.return_value()

    def return_value(self):
        return_dict = {}
        for k, v in self.values.items():
            return_dict[k] = v[self.current]
        return return_dict

if __name__ == "__main__":
    print("Launching app")
    app = QApplication(sys.argv)
    w = AppWindow()
    w.show()
    sys.exit(app.exec_())
