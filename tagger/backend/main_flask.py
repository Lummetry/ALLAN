from libraries.model_server.simple_model_server import SimpleFlaskModelServer

class FakeLogger:
  def __init__(self, **kwargs):
    return
  
  def P(self, s):
    print("LOG: " + s, flush=True)


class FakeModel(object):
  def __init__(self, logger, fn_model):
    self.logger = logger
    self.fn_model = fn_model
  
  def predict(self, usr_input):
    self.logger.P('Predicting on usr_input: {}'.format(usr_input))
    usr_input = str(usr_input) + ' PREDICTED'
    
    return usr_input
  
if __name__ == '__main__':
  l = FakeLogger(lib_name='MSRVT', no_folders_no_save=True) 
  dummy_model = FakeModel(l, 'path/to/model.h5')
  #callback input
  def dummy_inp_proc(data):
    if 'input_value' not in data.keys():
      print("ERROR: input json does not contain data")
      return None
  
    s = data['input_value']      
  
    return s
  
  #callback output
  def dummy_out_proc(data):
    return {'output_value':str(data)}
  
  dummy_server = SimpleFlaskModelServer(model=dummy_model, 
                               fun_input=dummy_inp_proc, 
                               fun_output=dummy_out_proc,
                               log=l,
                               port=5001)
  
  dummy_server.run()