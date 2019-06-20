
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:21:30 2017

@module:  Lummetry.AI SKU (Swiss Knife Utility)

@description:
    utility module
"""
 
__VER__ = '4.5.1.12'

from datetime import datetime as dt
import sys
import os
import socket
import pickle

from imageio import save as imsave

from io import TextIOWrapper, BytesIO
import numpy as np
import itertools
import json
from time import time as tm

import pandas as pd

from collections import OrderedDict

import textwrap

import imp

import codecs

from sklearn.metrics import r2_score, recall_score, precision_score, accuracy_score, f1_score

_HTML_START = "<HEAD><meta http-equiv='refresh' content='5' ></HEAD><BODY><pre>"
_HTML_END = "</pre></BODY>"

class Logger():
  """
  
  """
  def __init__(self, lib_name = "LOGR", lib_ver = "",
               config_file = "",
               base_folder = "", 
               DEBUG = True, 
               SHOW_TIME = True, 
               TF_KERAS = True,
               HTML = False,
               max_lines=1000,
               config_file_encoding=None
               ):
    """
     HTML:  True if output file will be HTML
     TF_KERAS: True if logger is Tensorflow/Keras aware
     DEBUG flag affects timer dictionary operations
    """
    self.devices  = {}
    self.max_lines = max_lines
    self.split_part = 1
    self.train_history = None
    self.DEBUG = DEBUG
    self.HTML = HTML
    self.timers = OrderedDict()
    self.app_log = list()
    self.results = list()
    self.printed = list()
    self.TF_KERAS = TF_KERAS
    self.KERAS_CONVERTER = True
    self.TF = False
    self.MACHINE_NAME = self.GetMachineName()
    self.__version__ = __VER__
    self.SHOW_TIME = SHOW_TIME
    self.last_time = tm()
    self.file_prefix = dt.now().strftime("%Y%m%d_%H%M%S")
    self.log_suffix = lib_name
    self.log_results_file = self.file_prefix + "_RESULTS.txt"
    self.__lib__= lib_name
    self._base_folder  = base_folder
    self.config_data = None
    self.watch_model = None

    self._configure_data_and_dirs(config_file, config_file_encoding)
    self._generate_log_path()
    self.log_results_file = os.path.join(self._logs_dir, self.log_results_file)
    
    ver = "v.{}".format(lib_ver) if lib_ver != "" else ""
    self.VerboseLog("Library [{} {}] initialized on machine [{}]".format(
                  self.__lib__, ver, self.MACHINE_NAME))
    self.VerboseLog("Logger version: {}".format(self.__version__))
    if self.TF_KERAS:
      self.CheckTF()
    self.VerboseLog("-------------------------")
    return
  
  
  def UpdateConfig(self, dict_newdata=None):
    """
     saves config file with current config_data dictionary
    """
    if dict_newdata is not None:
      for key in dict_newdata:
        self.config_data[key] = dict_newdata[key]
    with open(self.config_file, 'w') as fp:
        json.dump(self.config_data, fp, sort_keys=True, indent=4)         
    self.P("Config file '{}' has been updated.".format(self.config_file))
    return
  
  
  def GetConfigValue(self, key, default=0):
    if key in self.config_data.keys():
      _val = self.config_data[key]
    else:
      # create key if does not exist
      _val = default
      self.config_data[key] = _val
    return _val    
  
  
  def GetConfigData(self, key, default=0):
    return self.GetConfigValue(key, default)
    

  def _configure_data_and_dirs(self, config_file, config_file_encoding):
    if config_file != "":
      if config_file_encoding is None:
        f = open(config_file)
      else:
        f = open(config_file, encoding=config_file_encoding)
      self.config_data = json.load(f, object_pairs_hook=OrderedDict)
      new_dict = OrderedDict()
      for key in self.config_data.keys():
        new_dict[key.upper()] = self.config_data[key]
      self.config_data = new_dict
      assert ("BASE_FOLDER" in self.config_data.keys())
      assert ("APP_FOLDER" in self.config_data.keys())
      base_folder = self.config_data["BASE_FOLDER"]
      app_folder = self.config_data["APP_FOLDER"]
      if "GOOGLE" in base_folder.upper():
        base_folder = self.GetGoogleDrive()
      if "DROPBOX" in base_folder.upper():
        base_folder = self.GetDropboxDrive()
      self._base_folder  = os.path.join(base_folder,app_folder)
      print("Loaded config [{}]  BASE: {}".format(
          config_file,self._base_folder), flush = True)
      self.config_file = config_file
    else:
      self.config_data = {}
      self.config_data['BASE_FOLDER'] = '.'
      self.config_data['APP_FOLDER'] = '.'
      self.config_file = "default_config.txt"

    self._logs_dir = os.path.join(self._base_folder,"_logs")
    self._outp_dir = os.path.join(self._base_folder,"_output")
    self._data_dir = os.path.join(self._base_folder,"_data")
    self._modl_dir = os.path.join(self._base_folder,"_models")

    self._setup_folders([self._outp_dir, self._logs_dir, self._data_dir,
                         self._modl_dir])
  
  
  def _generate_log_path(self):
    part = '{:03d}'.format(self.split_part)
    lp = self.file_prefix
    ls = self.log_suffix
    if self.HTML:
      self.log_file = lp + '_' + ls + '_' + part +'_log_web.html'
    else:
      self.log_file = lp + '_' + ls + '_' + part + '_log.txt'
      
    self.log_file = os.path.join(self._logs_dir, self.log_file)
    path_dict = {}
    path_dict['CURRENT_LOG'] = self.log_file
    file_path = os.path.join(self._logs_dir, self.__lib__+'.txt')
    with open(file_path, 'w') as fp:
        json.dump(path_dict, fp, sort_keys=True, indent=4)         
    self._add_log("{} log changed to {}...".format(file_path, self.log_file))    
    return
  
  def _check_log_size(self):
    if self.HTML and (len(self.app_log) >= self.max_lines):
      self._add_log("Ending log part {}".format(self.split_part))
      self._save_log()
      self.app_log = []
      self.split_part += 1
      self._generate_log_path()
      self._add_log("Starting log part {}".format(self.split_part))
      self._save_log()
    return
  
  def CheckFolder(self, sub_folder):
    sfolder = os.path.join(self.GetBaseFolder(),sub_folder)
    if sfolder not in self.folder_list:
      self.folder_list.append(sfolder)
      
    if not os.path.isdir(sfolder):
      self.VerboseLog(" Creating folder [...{}]".format(sfolder[-40:]))
      os.makedirs(sfolder)
    return sfolder
    
  def LoadDataJSON(self, fname):
    datafile = os.path.join(self._data_dir,fname)
    self.VerboseLog('Loading data json: {}'.format(datafile))
    with open(datafile) as f:
      data_json = json.load(f)      
    return data_json


  def SaveDataJSON(self, data_json, fname):
    datafile = os.path.join(self._data_dir,fname)
    self.VerboseLog('Saving data json: {}'.format(datafile))
    with open(datafile, 'w') as fp:
        json.dump(data_json, fp, sort_keys=True, indent=4)         
    return data_json

  def SaveOutputJSON(self, data_json, fname):
    datafile = os.path.join(self._outp_dir,fname)
    self.VerboseLog('Saving output json: {}'.format(datafile))
    with open(datafile, 'w') as fp:
        json.dump(data_json, fp, sort_keys=True, indent=4)         
    return data_json

  
  def GetBaseFolder(self):
    return self._base_folder

  def GetDataFolder(self):
    return self._data_dir

  def GetOutputFolder(self):
    return self._outp_dir
  
  def GetOutputFile(self, fn):
    return os.path.join(self.GetOutputFolder(), fn)
  
  def GetModelsFolder(self):
    return self._modl_dir
  
  def GetPathFromNode(self, dct):
    if 'PARENT' in dct:
      path = self.GetPathFromNode(dct['PARENT'])
      os.path.join(path, dct['PATH'])
      return path
    elif 'USE_DROPBOX' in dct and int(dct['USE_DROPBOX']) == 1:
      return os.path.join(self.GetBaseFolder(), dct['PATH'])
    else:
      return dct['PATH']

    raise ValueError('Could not build path')
  
  def GetRootFile(self, str_file):
    fn = os.path.join(self._base_folder, str_file)
    assert os.path.isfile(fn)
    return fn
  
  
  def GetFileFromFolder(self, s_folder, s_file):
    s_fn = os.path.join(self.GetBaseFolder(),s_folder,s_file)
    if not os.path.isfile(s_fn):
      s_fn = None
    return s_fn
  
  def GetDataFile(self, s_file):
    """
    returns full path of a data file or none is file does not exist
    """
    fpath = os.path.join(self._data_dir, s_file)
    if not os.path.isfile(fpath):
      fpath = None
    return fpath

  def ModelExists(self, model_file):
    """
    returns true if model_file (check both .pb or .h5) exists
    """
    exists = False
    for ext in ['','.h5', '.pb']:
      fpath = os.path.join(self.GetModelsFolder(), model_file + ext)
      if os.path.isfile(fpath):        
        exists = True
    if exists:
      self.P("Detected model {}.".format(fpath))
    else:
      self.P("Model {} NOT found.".format(model_file))
    return exists
  
  def ReadFromPath(self, path):
    from os.path import splitext
    file_name, extension = splitext(path)
    if extension == '.csv':
      self.P('Reading from {}'.format(path))
      df = pd.read_csv(path)
      self.P('Done reading from {}'.format(path), show_time=True)
      return df
    elif extension == '.xls' or extension == '.xlsx':
      self.P('Reading from {}'.format(path))
      df = pd.read_excel(path)
      self.P('Done reading from {}'.format(path), show_time=True)
      return df
    elif extension == '.pkl':
      self.P('Reading from {}'.format(path))
      with open(path, 'rb') as handle:
        df = pickle.load(handle)
      self.P('Done reading from {}'.format(path), show_time=True)
      return df    
    raise ValueError('Extension {} not understood!'.format(extension))
  
  def WriteToPath(self, path, data):
    from os.path import splitext
    file_name, extension = splitext(path)
    if extension == '.csv':
      if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
      data.to_csv(path, index=False)
    elif extension == '.xls' or extension == '.xlsx':
      if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
      data.to_excel(path, index=False)
    elif extension == '.pkl':
      with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
  def SetNicePrints(self, precision=3):
    np.set_printoptions(precision=precision)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)    
    pd.options.display.precision = precision
    return  
  
  
  def GetOutputFile(self, s_file):
    fpath = os.path.join(self._outp_dir, s_file)
    if not os.path.isfile(fpath):
      fpath = None
    return fpath
    
   
  def GetGoogleDrive(self):
    home_dir = os.path.expanduser("~")
    valid_paths = [
                   os.path.join(home_dir, "Google Drive"),
                   os.path.join(home_dir, "GoogleDrive"),
                   os.path.join(os.path.join(home_dir, "Desktop"), "Google Drive"),
                   os.path.join(os.path.join(home_dir, "Desktop"), "GoogleDrive"),
                   os.path.join("C:/", "GoogleDrive"),
                   os.path.join("C:/", "Google Drive"),
                   os.path.join("D:/", "GoogleDrive"),
                   os.path.join("D:/", "Google Drive"),
                   ]
  
    drive_path = None
    for path in valid_paths:
      if os.path.isdir(path):
        drive_path = path
        break

    if drive_path is None:
      raise Exception("Couldn't find google drive folder!")    

    return drive_path  


  def GetDropboxDrive(self):
    home_dir = os.path.expanduser("~")
    valid_paths = [
                   os.path.join(home_dir, "Lummetry.AI Dropbox/DATA"),
                   os.path.join(home_dir, "Lummetry.AIDropbox/DATA"),
                   os.path.join(os.path.join(home_dir, "Desktop"), "Lummetry.AI Dropbox/DATA"),
                   os.path.join(os.path.join(home_dir, "Desktop"), "Lummetry.AIDropbox/DATA"),
                   os.path.join("C:/", "Lummetry.AI Dropbox/DATA"),
                   os.path.join("C:/", "Lummetry.AIDropbox/DATA"),
                   os.path.join("D:/", "Lummetry.AI Dropbox/DATA"),
                   os.path.join("D:/", "Lummetry.AIDropbox/DATA"),
                   ]
  
    drive_path = None
    for path in valid_paths:
      if os.path.isdir(path):
        drive_path = path
        break

    if drive_path is None:
      raise Exception("Couldn't find google drive folder!")    

    return drive_path  
      
  
  def _setup_folders(self,folder_list):
    self.folder_list = folder_list
    for folder in folder_list:
      if not os.path.isdir(folder):
        print("Creating folder [{}]".format(folder))
        os.makedirs(folder)
    return

  def ShowNotPrinted(self):
    nr_log = len(self.app_log)
    for i in range(nr_log):
      if not self.printed[i]:
        print(self.app_log[i], flush = True)
        self.printed[i] = True
    return
  
  
 
  def _logger(self, logstr, show = True, results = False, noprefix = False, show_time = False):
    """
    log processing method
    """
    elapsed = tm() - self.last_time
    
    self._add_log(logstr, show = show, results = results, noprefix = noprefix, show_time = show_time)
    
    self._save_log()
    
    self.last_time = tm()
    
    self._check_log_size()
    return elapsed
  
  
  
  def _add_log(self, logstr, show = True, results = False, noprefix = False, show_time = False):
    if logstr == "":
      logstr = " "
    elapsed = tm() - self.last_time
    nowtime = dt.now()
    prefix = ""
    strnowtime = nowtime.strftime("[{}][%Y-%m-%d %H:%M:%S] ".format(self.__lib__))
    if self.SHOW_TIME and (not noprefix):
      prefix = strnowtime
    if logstr[0]=="\n":
      logstr = logstr[1:]
      prefix = "\n"+prefix
    logstr = prefix + logstr
    if show_time:
      logstr += " [{:.2f}s]".format(elapsed)
    self.app_log.append(logstr)
    if show:
      print(logstr, flush = True)
      self.printed.append(True)
    else:
      self.printed.append(False)
    if results:
      self.results.append(logstr)
    return
  
  
  
  def _save_log(self):
    nowtime = dt.now()
    strnowtime = nowtime.strftime("[{}][%Y-%m-%d %H:%M:%S] ".format(self.__lib__))
    stage = 0
    try:
      log_output = codecs.open(self.log_file, "w", "utf-8")  #open(self.log_file, 'w+')
      stage += 1
      if self.HTML:
        log_output.write(_HTML_START)
        stage += 1
        iter_list = reversed(self.app_log)
      else:
        iter_list = self.app_log
      for log_item in iter_list:
        #if self.HTML:
        #  log_output.write("%s<BR>\n" % log_item)
        #else:
          log_output.write("{}\n".format(log_item))
          stage += 1
      if self.HTML:
        log_output.write(_HTML_END)
        stage += 1
      log_output.close()
      stage += 1
    except:
      print(strnowtime+"LogWErr S: {} [{}]".format(stage,
            sys.exc_info()[0]), flush = True)
    return
  

  def VerboseLog(self,str_msg, results=False, show_time=False, noprefix = False):    
    return self._logger(str_msg, show = True, results = results, show_time = show_time,
                        noprefix = noprefix)
  
  def P(self,str_msg, results = False, show_time = False, noprefix = False):    
    return self._logger(str_msg, show = True, results = results, show_time = show_time,
                        noprefix = noprefix)
  
  def PrintPad(self, str_msg, str_text, n=3):
    str_final = str_msg + "\n" +  textwrap.indent(str_text, n * " ")
    self._logger(str_final, show = True, results = False, show_time = False)
    return

  def Log(self,str_msg, show = False, results = False, show_time = False):    
    return self._logger(str_msg, show = show, results = results, show_time = show_time )

  
  # custom_objects: dict with 'custom_name': custom_function
  def LoadKerasModel(self, model_name, custom_objects = None, DEBUG=True, use_gdrive=True):
    if model_name[-3:] != '.h5':
      model_name += '.h5'
    if DEBUG: self.VerboseLog("Trying to load {}...".format(model_name))
    if use_gdrive is True:
      model_full_path = os.path.join(self.GetModelsFolder(), model_name)
    else:
      model_full_path = model_name

    if os.path.isfile(model_full_path):
      from keras.models import load_model
      if DEBUG: self.VerboseLog("Loading [...{}]".format(model_full_path[-40:]))
      model = load_model(model_full_path, custom_objects = custom_objects)
      if DEBUG: self.VerboseLog("Done loading [...{}]".format(model_full_path[-40:]), show_time = True)
    else:
      self.VerboseLog("File {} not found.".format(model_full_path))
      model = None
    return model


  def GetMachineName(self):
    """
    if socket.gethostname().find('.')>=0:
        name=socket.gethostname()
    else:
        name=socket.gethostbyaddr(socket.gethostname())[0]
    """
    self.MACHINE_NAME = socket.gethostname()
    return self.MACHINE_NAME

  def _check_keras_avail(self):
    try:
        imp.find_module('keras')
        found = True
        import keras
        self.KERAS_VER = keras.__version__
        self.KERAS = True
    except ImportError:
        found = False
        self.KERAS = False
    return found
    

  def _check_tf_avail(self):
    try:
        imp.find_module('tensorflow')
        found = True
        import tensorflow as tf
        self.TF_VER = tf.__version__
        self.TF = True
    except ImportError:
        found = False
        self.TF = False
    return found

  def CheckTF(self):
    ret = 0
    if self._check_tf_avail():
      self.TF = True
      from tensorflow.python.client import device_lib
      local_device_protos = device_lib.list_local_devices()
      self.devices ={x.name: {'name':x.physical_device_desc,
                              'mem':x.memory_limit,}
                              for x in local_device_protos}
      types = [x.device_type for x in local_device_protos]
      self.gpu_mem = []
      if 'GPU' in types:
          ret = 2
          self._logger("Found TF {} running on GPU".format(self.TF_VER))
          for _dev in self.devices:
            if 'GPU' in _dev.upper():
              self._logger(" {}:".format(_dev[-5:]))
              self._logger("  Name: {}".format(self.devices[_dev]["name"]))
              self.gpu_mem.append(self.devices[_dev]["mem"] / (1024**3))
              self._logger("  Mem:  {:.1f} GB".format(self.gpu_mem[-1]))
      else:
          self._logger("Found TF {} running on CPU".format(self.TF_VER))
          ret = 1
      try:
        import tensorflow as tf
        self.TF_KERAS_VER = tf.keras.__version__
        self._logger("Found TF.Keras {}".format(self.TF_KERAS_VER))
      except:
        self.TF_KERAS_VER = None
        self._logger("No TF.Keras found.")
        
      if self._check_keras_avail():
        self._logger("Found Keras {}".format(self.KERAS_VER))
    else:
      self._logger("TF not found")
      self.TF = False
    return ret
  
  
  def GetGPU(self):
    res = []
    if self._check_tf_avail():
      self.TF = True
      from tensorflow.python.client import device_lib
      loc = device_lib.list_local_devices()
      res = [x.physical_device_desc for x in loc if x.device_type=='GPU']
    return res
      

  def start_timer(self, sname):
    if not self.DEBUG:
      return -1

    count_key = sname+"___COUNT"
    start_key = sname+"___START"
    pass_key  = sname+"___PASS"
    if not (count_key in self.timers.keys()):
      self.timers[count_key] = 0
      self.timers[sname] = 0
      self.timers[pass_key] = True
    ctime = tm()
    self.timers[start_key] = ctime
    return ctime


  def end_timer(self, sname, skip_first_timing = True):
    result = 0
    if self.DEBUG:
      count_key = sname+"___COUNT"
      start_key = sname+"___START"
      end_key   = sname+"___END"
      pass_key  = sname+"___PASS"
      
      self.timers[end_key] = tm()
      result = self.timers[end_key] - self.timers[start_key]
      _count = self.timers[count_key]
      _prev_avg = self.timers[sname]
      avg =  _count *  _prev_avg
      
      if self.timers[pass_key] and skip_first_timing:
        self.timers[pass_key] = False
        return result # do not record first timing in average
      
      self.timers[count_key] = _count + 1
      avg += result
      avg = avg / self.timers[sname+"___COUNT"]
      self.timers[sname] = avg
    return result
  
  def show_timers(self):
    if self.DEBUG:
      self.VerboseLog("Timing results:")
      for key,val in self.timers.items():
        if not ("___" in key):
          self.VerboseLog(" {} = {:.3f}s".format(key,val))
    else:
      self.VerboseLog("DEBUG not activated!")
    return
  
  def get_stats(self):
    self.show_timers()
    return
  def show_timings(self):
    self.show_timers()
    return
  def get_timing(self, skey):
    return self.timers[skey]