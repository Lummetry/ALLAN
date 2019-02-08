import numpy as np
import pickle

class ChatBot:
  def __init__(self, logger):
    self.logger = logger
    self.config_data = self.logger.config_data
    self._parse_config_data()
    return


  def _parse_config_data(self):
    self.encoder_name = self.config_data['ENCODER_NAME']
    self.decoder_name = self.config_data['DECODER_NAME']
    self.decoder_architecture = self.config_data['DECODER']

    fn_labels_dictionary = self.logger.GetDataFile(self.config_data['LABELS_DICTIONARY'])
    fn_inv_labels_dictionary = self.logger.GetDataFile(self.config_data['INV_LABELS_DICTIONARY'])

    with open(fn_labels_dictionary, 'rb') as handle:
      self.labels_dictionary = pickle.load(handle)
    with open(fn_inv_labels_dictionary, 'rb') as handle:
      self.inv_labels_dictionary = pickle.load(handle)

    return


  def _log(self, str_msg, results = False, show_time = False, noprefix=False):
    self.logger.VerboseLog(str_msg, results=results, show_time=show_time,
                           noprefix=noprefix)
    return


  def LoadModels(self):
    self.enc_pred_model = self.logger.LoadKerasModel(self.encoder_name, use_gdrive=True)
    self.dec_pred_model = self.logger.LoadKerasModel(self.decoder_name, use_gdrive=True)

    l = list(map(lambda x: x.name, self.enc_pred_model.layers))
    self.enc_recurrent_layers = list(filter(lambda x: 'rec' in x, l))

    l = list(map(lambda x: x.name, self.dec_pred_model.layers))
    self.dec_recurrent_layers = list(filter(lambda x: 'rec' in x, l))

    self.dec_recurrent_units = []
    for l in self.dec_pred_model.layers:
      if 'rec' in l.name:
        self.dec_recurrent_units.append(l.units)

    return


  def organize_text(self, text):
    text = text[1:]
    text = text.replace(' ?', '?')
    text = text.replace(' !', '!')
    text = text.replace(' ,', ',')
    text = text.replace(' .', '.')
    return text

  def _step_by_step_prediction(self, data_loader, _input, method='sampling', verbose=1):
    assert method in ['sampling', 'argmax']

    str_input = _input
    if type(str_input) is list: str_input = '\n'.join(str_input)

    input_tokens = data_loader.input_word_text_to_tokens(str_input, use_characters=True)
    input_tokens = np.expand_dims(np.array(input_tokens), axis=0)

    if verbose: self._log("Given '{}' the decoder predicted:".format(str_input))
    predict_results = self.enc_pred_model.predict(input_tokens)
    enc_states = predict_results[:-1]
    label = np.argmax(predict_results[-1][:,[-1],:], axis=-1)
    dec_model_inputs = []

    for i, enc_l in enumerate(self.decoder_architecture['INITIAL_STATE']):
      units = self.dec_recurrent_units[i]

      inp_h = np.zeros((1, units))
      inp_c = np.zeros((1, units))

      if enc_l != "":
        idx = self.enc_recurrent_layers.index(enc_l)
        if idx is not None:
          inp_h, inp_c = enc_states[2*idx:2*(idx+1)]
          self._log("Enc_h: {}  Dec_h: {}".format(inp_h.sum(), inp_c.sum()))
      #endif

      dec_model_inputs += [inp_h, inp_c]
    #endfor

    current_gen_token = data_loader.start_char_id
    predicted_tokens = []
    nr_preds = 0
    while current_gen_token != data_loader.end_char_id:
      current_gen_token = np.array(current_gen_token).reshape((1,1))
      dec_model_outputs = self.dec_pred_model.predict(dec_model_inputs + [current_gen_token, label])

      P = dec_model_outputs[-1].squeeze()
      if method == 'sampling':
        current_gen_token = np.random.choice(range(P.shape[0]), p=P)
      if method == 'argmax':
        current_gen_token = np.argmax(P)
  
      predicted_tokens.append(current_gen_token)
      dec_model_inputs = dec_model_outputs[:-1]
      
      nr_preds += 1
      if nr_preds == 50:
        break
    #end_while
    predicted_tokens = predicted_tokens[:-1]
    predicted_text = data_loader.input_word_tokens_to_text([predicted_tokens])
    predicted_text = self.organize_text(predicted_text)

    if verbose:
      self._log("  --> '{}'".format(predicted_text))
    return predicted_text, self.inv_labels_dictionary[label]