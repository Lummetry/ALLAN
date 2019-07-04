import os
import pickle
import string
import random
import requests

from libraries.logger import Logger
from time import time, sleep

from bokeh.layouts import column, layout, row, Spacer, widgetbox
from bokeh.models import Div, Paragraph, TextInput, ColumnDataSource, Dropdown
from bokeh.plotting import figure
from nltk.tokenize import word_tokenize

from wordcloud import WordCloud  


class Server:
    """ Put together everything in the layouts """

    def _log(self, str_msg, results=False, show_time=False):
        """ Helper function for logging """
        self.logger.VerboseLog(str_msg, results, show_time)

    def __init__(self, bot_name, show_bot_messages=True, config_file='config_runner.txt'):
        self.show_bot_messages = show_bot_messages
        self.bot_name = bot_name
        self.bot_name_placeholder = '<NAMEBOT>'
        self.URL_create_conv = "http://127.0.0.1:8000/api_create_conversation/"
        self.URL_create_msg  = "http://127.0.0.1:8000/api_create_message/"
        
        self.bd_chat_id = {}
        
        self.tasteaza_msg = "{} tasteaza ...".format(self.bot_name)
        if not self.show_bot_messages: self.tasteaza_msg = "{} raspunde ...".format(self.bot_name)

        self.CONV_STATE = None  # 0 - System (Name); 1 (Oana)
        self.CONV_PREFIX = ''.join(
            random.choice(string.ascii_uppercase + string.digits + string.ascii_lowercase) for _ in range(15))
        self.USERNAME = None
        self.custom_delay = False
        self.message_history = []
        self.user_message_history = []
        self.CONFIG = None
        self.timeout = 60
        self.FLAG_TIMEOUT = False
        self.FLAG_CONVERSATION = False
        self.logger = Logger(lib_name="CHATBOT", config_file=config_file,
                             TF_KERAS=False,
                             HTML=True)
        self.profile = []


        self._log('Initializing server {}...'.format(self.CONV_PREFIX))

        self.current_lines = None
        self.current_page = -1
        self.CONFIG = self.logger.config_data

        with open(self.CONFIG['VULGARITIES'], 'rt', encoding='utf-8') as handle:
          self.vulgarities = handle.read().splitlines()


        self.dct_domain_id = {'medical': 1, 'imobiliare': 2}
        self.bot_type = self.CONFIG['BOT_TYPE']
        
        for domain,domain_id in self.dct_domain_id.items():
          r = requests.post(url=self.URL_create_conv, data={'domain_id': domain_id})
          self.bd_chat_id[domain_id] = r.json()['data']['id']
        #endfor
    
        
        if self.bot_type == 'imobiliare':
          intro_messages = [
               'Bine ai venit! Ma numesc {} si sunt aici sa te ajut. Ce te-ar interesa?',
               'Bine ai venit! Ma numesc {}! Cu ce te pot ajuta?',
               'Bine ai venit! Ma numesc {}! Cum iti pot fi de folos?',
               'Bine ai venit! Ma numesc {}! De ce informatii ai nevoie?',
               'Bine ai venit! Ma numesc {}. Ce te-ai interesa?',
               'Bine ai venit! Ma numesc {}. Ce te-ar interesa?',
               'Bine ai venit! Ma numesc {}. Cum te pot ajuta?',
               'Bine ai venit! numele meu este {}! Cu ce informatii te pot ajuta?',
               'Bun venit! Numele meu este {}! Cu ce te pot ajuta?',
               'Bun venit! Numele meu este {}! Cum te pot ajuta?',
               'Buna ziua! Ma numesc {}! Cum pot sa te ajut?',
               'Buna ziua! Ma numesc {}, cu ce te pot ajuta?',
               'Buna ziua! Ma numesc {}, cum te pot ajuta azi?',
               'Buna ziua! Numele meu este {} si sunt aici sa te ajut!',
               'Buna ziua! Numele meu este {} si sunt aici sa te ajut.',
               'Buna ziua! Numele meu este {}, te pot ajuta cu ceva?',
               'Buna ziua! Numele meu este {}. Cu ce informatii te pot ajuta astazi?',
               'Buna ziua! Numele meu este {}. Cu ce informatii te pot ajuta?',
               'Buna ziua! Sunt {}, cu ce te pot ajuta astazi?',
               'Buna ziua! Sunt {}. Cu ce as putea sa te ajut?',
               'Buna ziua! Sunt {}. Cu ce informatii te pot ajuta?'
          ]
        
        elif self.bot_type == 'medical':
          intro_messages = [
               'Bine ai venit! Ma numesc {}. Cu ce informatii te pot ajuta?',
               'Bine ai venit! Ma numesc {}. Te pot ajuta cu ceva?',
               'Bine ai venit! Numele meu este {} si m-as bucura sa te pot ajuta cu ceva.',
               'Bine ai venit! Numele meu este {} si m-as bucura sa te pot ajuta.',
               'Bine ai venit! Numele meu este {}. Cu ce informatii te pot ajuta?',
               'Bine ai venit! Numele meu este {}. Cu ce te pot ajuta?',
               'Buna ziua! {} ma numesc si sunt aici sa te ajut.',
               'Buna ziua! Eu sunt {}. Cu ce te pot ajuta?',
               'Buna ziua! Eu sunt {}. Cum pot sa te ajut azi?',
               'Buna ziua! Eu sunt {}. Te pot ajuta cu ceva?',
               'Buna ziua! Ma cheama {}. Cum te pot ajuta?',
               'Buna ziua! Ma numes {}, cu informatii te pot ajuta?',
               'Buna ziua! Ma numesc {} si sunt aici sa te ajut.',
               'Buna ziua! Ma numesc {}! Te pot ajuta cu ceva?',
               'Buna ziua! Ma numesc {}, te pot ajuta cu ceva?',
               'Buna ziua! Ma numesc {}. Care este problema dvs?',
               'Buna ziua! Ma numesc {}. Cu ce informatii te pot ajuta azi?',
               'Buna ziua! Ma numesc {}. Cu ce informatii te pot ajuta?',
               'Buna ziua! Ma numesc {}. Cu ce te pot ajuta, <NAME>?',
               'Buna ziua! Ma numesc {}. Cu ce te pot ajuta?',
               'Buna ziua! Ma numesc {}. Cum te pot ajuta astazi?',
               'Buna ziua! Ma numesc {}. Cum te pot ajuta?',
               'Buna ziua! Numele meu este {} si sunt aici sa te ajut.',
               'Buna ziua! Numele meu este {}! Cu ce informatii te pot ajuta?',
               'Buna ziua! Numele meu este {}, cu ce te pot ajuta?',
               'Buna ziua! Numele meu este {}. Cu ce informatii te pot ajuta astazi?',
               'Buna ziua! Sunt {}. Cu ce te pot ajuta astazi?',
               'Buna ziua! Sunt {}. Cu ce te pot ajuta?',
               'Buna ziua! Sunt {}. Sunt aici sa te ajut.',
               'Buna ziua, <NAME>! Sunt {}. Cum iti pot fi de folos?',
               'Buna ziua, eu ma numesc {}! Care este problema?',
               'Buna! Eu sunt {}. Cu ce informatii te pot ajuta?',
               'Buna! Eu sunt {}. Cu ce te pot ajuta, <NAME>?',
               'Buna! Eu sunt {}. Cu ce te pot ajuta?',
               'Buna! Ma numesc {}, te pot ajuta cu ceva?',
               'Buna! Ma numesc {}. Cu ce te pot ajuta?',
               'Buna! Ma numesc {}. Cum iti pot fi de folos?',
               'Buna! Numele meu este {} si sunt aici sa te ajut.',
               'Buna! Numele meu este {}. Cu ce te pot ajuta?',
               'Buna! Numele meu este {}. In ce fel as putea sa te ajut?',
               'Buna! Numele meu este {}. Te pot ajuta cu ceva?',
               'Buna! Sunt {}. Cu ce te pot ajuta?',
               'Buna, ma numesc {}! Cu ce te pot ajuta?',
               'Buna, ma numesc {}! Te pot ajuta cu ceva?',
               'Buna, numele meu este {}! Cu ce te pot ajuta?',
               'Buna. Sunt {}. Cu ce iti pot fi de folos azi?',
               'Buna. Sunt {}. Cu ce te pot ajuta?',
               'Hei, ma numesc {}, te pot ajuta cu ceva?',
               'Ma numesc {}. Cu ce te pot ajuta astazi?',
          ]
        

        for i in range(len(intro_messages)):
          intro_messages[i] = intro_messages[i].format(self.bot_name)

        self.intro_message = random.choice(intro_messages)
        self.message_history.append(self.intro_message)
        self.push_msgs_to_db(msgs=[(self.intro_message.replace(self.bot_name, self.bot_name_placeholder), 'salut', False)],
                             domain_id=self.dct_domain_id[self.bot_type])

        self.reply_labels, self.max_len_labels = self.get_labels()
        self.reply_hashtags = self.get_hashtags()

        txt = """
            <div class="container pull-left message ml-4">
                <div class="row pull-left text-muted text-small">{}</div><br>
        """

        self.reply_text = txt.format(self.bot_name) +\
          """<div class="row pull-left bg-reply p-2 mr-4 mb-2 text-white rounded chat-msg">{}</div></div>"""
        self.reply_text_system = txt.format("Mesaj de sistem") +\
          """<div class="row pull-left bg-system p-2 mr-4 mb-2 text-white rounded chat-msg">{}</div></div>"""
        self.is_typing_text = txt.format(self.bot_name) +\
          """<div class="row pull-left bg-typing p-2 mr-4 mb-2 text-white rounded chat-msg">{}</div></div>"""
        
        self.user_message_text = """
            <div class="container pull-right message mr-4">
                    {1}        
                    <div class="col">
                        <p class="row pull-right text-muted text-small">Tu</p><br>
                        <p class="row pull-right bg-user-message p-2 ml-4 text-white rounded chat-msg">{0}</p>
                    </div>
                      
            </div>
        """

        self.label_text = """
            <div class="col">
                <p class="row pull-right bg-label-message p-2 lg-ml-4 sm-mt-4 rounded keep-spaces font-courier">{}</p>
            </div>
        """

        self.hashtag_text = """
            <p class="h2 labels-profile">#{}</p>
        """

        # default_text = '<p class="pull-left bg-secondary p-2 ml-3 mt-3 text-white rounded">{}</p><br><br>'.format(intro_message)
        default_text = ""
        # self.conversation_height = 600
        # self.conversation_width = 900

        self.wordcloud = WordCloud(background_color='white',
                                   mode='RGBA',
                                   width=350,
                                   height=250,
                                   regexp="[#\w][\w']+")

        wordcloud_img_name = 'wc_{}.png'
        self.wordcloud_img_path = 'static/img/wordcloud/' + wordcloud_img_name
        self.wordcloud_img_full_path = 'allan_web_platform/' + self.wordcloud_img_path

        if not os.path.exists(os.path.dirname(self.wordcloud_img_path)):
            os.mkdir(os.path.dirname(self.wordcloud_img_path))

        self.wordcloud_image = Div(text='', height=300, width=300)

        # self.wordcloud.generate(' '.join(['ce', 'face', 'ce'])) \
        #     .to_file(self.wordcloud_img_path.format(len(self.user_message_history)))

        # self.wordcloud_image.text = '<img src="{}">'.format(
        #     self.wordcloud_img_full_path.format(len(self.user_message_history)))

        self.div_hashtags = Div(text='', css_classes=['scrollable', 'bg-hashtags', 'rounded', 'p-3'], height=450,
                                width=350)
        self.div_conversation = Div(text=default_text, css_classes=['scrollable', 'bg-light', 'rounded'], height=600,
                                    width=800)

        self.p_hashtag_title = Paragraph(text='Profil',
                                         css_classes=['text-light', 'h4', 'font-weight-bold', 'text-center'])
        p_chat_title_text = 'TempRent'  # 'Conversatia dumneavoastra'
        if not self.show_bot_messages: p_chat_title_text = ''
        self.p_chat_title = Paragraph(text=p_chat_title_text,
                                      css_classes=['text-light', 'h4', 'font-weight-bold', 'text-center'])
        self.p_chat_footer = Paragraph(text='', css_classes=['text-light', 'h6', 'pt-2', 'text-right'])
        self.input_conversation = TextInput(placeholder="Tastaţi mesajul aici", width=855, css_classes=['mt-0', 'pt-0'])
        self.input_conversation.on_change('value', lambda attr, new, old: self._send_message())

        dropdown_menu = [('Imobiliare', 'imobiliar'), ('Medical', 'medical')]
        self.dd_menu_dict = dict([tuple(reversed(el)) for el in dropdown_menu])
        # Use button_type default, primary, success, warning, danger to change its color
        self.dropdown = Dropdown(label=self.dd_menu_dict[self.bot_type],
                                 menu=dropdown_menu, width=100,
                                 button_type="default")
        self.dropdown.on_change('value', self._change_bot_type)
        if self.show_bot_messages: self.RequireName()
        else: self.CONV_STATE = 1

        self._log("  Finished initializing the server.", show_time=True)

    def RequireName(self):
        text = self.reply_text_system.format("Bine ai venit! Te rog sa introduci prenumele tau:")
        div_conversation_copy = self.div_conversation.text
        self.div_conversation.text = div_conversation_copy + text
        self.CONV_STATE = 0
        return
      
    def push_msgs_to_db(self, msgs, domain_id):
      for (turn, label, is_human) in msgs:
        api_msg = {'chat_id': self.bd_chat_id[domain_id],
                   'human': is_human,
                   'message': turn,
                   'label': label}
        r_msg = requests.post(url=self.URL_create_msg, data=api_msg)
        
        success = r_msg.json()['success']
        self._log("Msg ({},{},{}) post with success flag={}".format(turn,label,is_human,success))
        
      return
      
    def BanVulgarity(self, message):
        if len(set(message.lower().split()) & set(self.vulgarities)):
            self._log("Message {} banned due to vulgarity.".format(message))
            self.CONV_STATE = 2
        
        return

    def CreateLayout(self):
        """ Creates the layout of the app
        This function is called from the main file
        Returns a list of layouts """
        self._log('Creating the layout of the server ...')

        # layout_tab1 = column(widgetbox(self.p_chat_title,
        #                                row(self.div_hashtags, self.div_conversation),
        #                                self.input_conversation,
        #                                self.p_chat_footer,
        #                                height=600, width=955,
        #                                css_classes=['bg-success', 'card', 'card-body', 'p-3']))

        layout_tab1 = row(widgetbox(self.p_hashtag_title,
                                    self.div_hashtags,
                                    self.wordcloud_image,
                                    width=385, height=550,
                                    css_classes=['bg-main', 'card', 'card-body', 'p-3']
                                    ),
                          Spacer(width=40),
                          widgetbox(self.p_chat_title,
                                    self.div_conversation,
                                    self.input_conversation,
                                    #self.dropdown,
                                    self.p_chat_footer,
                                    width=833, height=600,
                                    css_classes=['bg-main', 'card', 'card-body', 'p-3']),
                          css_classes=['bg-transparent', 'm-4'], height=800)

        self._log('Created the layout of the server.', show_time=True)

        return [layout_tab1]

    def _change_bot_type(self, attr, old, new):
        """ Change bot type on dropdown value change """
        self.bot_type = new
        self.dropdown.label = self.dd_menu_dict[new]
        self._log(' Chat type changed to: {}'.format(new))

    def get_labels(self):
        with open(self.CONFIG['LABELS'], 'r') as f:
            labels = f.read().splitlines()

        max_len = len(max(labels, key=lambda x: len(x)))
        return labels, max_len

    def get_hashtags(self):
        with open(self.CONFIG['HASHTAGS'], 'r') as f:
            return f.read().splitlines()

    def _standard_end_message(self):
        if self.FLAG_CONVERSATION:
            return
        if self.FLAG_TIMEOUT:
            reply = "Inainte sa pleci, voiam sa iti recomand sa descarci voucher-ul pe care ti-l ofer intrand pe acest link ......"
            div_conversation_copy = self.div_conversation.text
            self.div_conversation.text = div_conversation_copy + '<br><br><p class="pull-left bg-secondary p-2 ml-3 mt-4 text-white rounded">{}</p><br><br>'.format(
                reply)
            self.FLAG_CONVERSATION = True
        self.FLAG_TIMEOUT = True
        return

    def _send_message(self):
        """ Submit user message """
        self.FLAG_TIMEOUT = False
        new_message = self.input_conversation.value.strip()

        self.input_conversation.value = ''
        self._log('New message received: [{}]'.format(new_message))

        if new_message == '':
            return

        self.BanVulgarity(new_message)

        if self.CONV_STATE == 1:
            self.message_history.append(new_message)
            # self.div_conversation.text += '<p class="pull-right bg-success p-2 mr-4 mt-4 text-white rounded">{}</p><br><br>'.format(new_message)
            div_conversation_copy = self.div_conversation.text
            self.div_conversation.text += self.user_message_text.format(new_message.replace('<UNK>', '[UNK]'), '')
            # self.div_conversation.text += '<p class="pull-left bg-secondary p-2 ml-3 mt-4 text-white rounded">Oana tastează...</p><br><br>'
            self.div_conversation.text += self.is_typing_text.format(self.tasteaza_msg)
            self._log('Got the user message')
            reply, label, is_hashtag = self._generate_reply(message=new_message, max_message_length=50,
                                                            custom_delay=self.custom_delay)
            self._log('Got the reply {} | {} | {}'.format(reply, label, is_hashtag))
            self.message_history.append(reply)
            
            self.push_msgs_to_db(msgs=[(new_message, label, True), (reply, 'neutru', False)],
                                 domain_id=self.dct_domain_id[self.bot_type])

            if False:
              if not os.path.exists('conversations'):
                  os.mkdir('conversations')
              with open('conversations/{}.txt'.format(self.CONV_PREFIX), 'wt') as f:
                  f.write('\n'.join(self.message_history))

            padded_labels = '[{}]'.format(label.strip()).rjust(self.max_len_labels)
            label_text = self.label_text.format(padded_labels)
            self.div_conversation.text = div_conversation_copy + self.user_message_text.format(
                new_message.replace('<UNK>', '[UNK]'), label_text)

            reply = reply.replace('<UNK>', '[UNK]')
            reply = reply.replace('<NAME>', self.USERNAME)

            self._log('Showing the reply')
            # self.div_conversation.text = div_conversation_copy + '<p class="pull-left bg-secondary p-2 ml-3 mt-4 text-white rounded">{}</p><br><br>'.format(reply)
            
            if self.show_bot_messages:
                self.div_conversation.text += self.reply_text.format(reply)
            if is_hashtag and label not in self.profile:
                self._log('is label so add it to hashtags')
                self.profile.append(label)
                self.div_hashtags.text += self.hashtag_text.format(label)

            # clean_message = ' '.join([word for word in word_tokenize(new_message) if word not in self.stopwords])

            self.user_message_history.append(new_message)
            if is_hashtag:
                if self.user_message_history.count('#{}'.format(label)) < 10:
                    print('label added')
                    self.user_message_history.extend(['#{}'.format(label)]*5)

            print(self.user_message_history)

            wordcloud_dir_files = os.path.dirname(self.wordcloud_img_path)
            if os.listdir(wordcloud_dir_files):
                # remove existing wordcloud images to save space
                for file in os.listdir(wordcloud_dir_files):
                    try:
                        os.remove(os.path.join(wordcloud_dir_files, file))
                    except:
                        self._log('     FILE {} COULD NOT BE DELETED'.format(file))

            self.wordcloud.generate(' '.join(self.user_message_history)) \
                .to_file(self.wordcloud_img_path.format(len(self.user_message_history)))

            self.wordcloud_image.text = '<img src="{}">'.format(
                self.wordcloud_img_full_path.format(len(self.user_message_history)))

        elif self.CONV_STATE == 0:
            self.div_conversation.text += self.user_message_text.format(new_message.replace('<UNK>', '[UNK]'), '')
            self.USERNAME = new_message
            self._log("Username: {}".format(self.USERNAME))

            self.CONV_STATE = 1
            self.div_conversation.text += self.reply_text.format(self.intro_message)
        elif self.CONV_STATE == 2:
            div_conversation_copy = self.div_conversation.text  
            self.div_conversation.text = div_conversation_copy + self.user_message_text.format(new_message.replace('<UNK>', '[UNK]'), '')
            
            text = self.reply_text_system.format("Ne pare rau, insa cat timp folosesti cuvinte vulgare, conversatia nu poate continua.")
            div_conversation_copy = self.div_conversation.text
            self.div_conversation.text = div_conversation_copy + text
            
            self.CONV_STATE = 1

    def _change_name_in_msg_hist(self):
      self.message_history = [m.replace(self.bot_name, self.bot_name_placeholder) for m in self.message_history]
      return

    def _generate_reply(self, message, max_message_length=100, custom_delay=False):
        """ Return the reply of a user message
        If custom_delay is True, it will wait before giving the reply back 
        to simulate human typing """

        if len(message.split()) > max_message_length:
            reply = "Scuze, mesajul este prea lung."
            label = ""
            is_hastag = False
        else:
            print(self.message_history)
            self._change_name_in_msg_hist()
            prefix = ''.join(
                random.choice(string.ascii_uppercase + string.digits + string.ascii_lowercase) for _ in range(15))

            with open('pipe/{}_message.pickle'.format(prefix), 'wb') as handle:
                pickle.dump(self.message_history, handle, protocol=pickle.HIGHEST_PROTOCOL)

            fn_response = 'pipe/{}_response.pickle'.format(prefix)
            while not os.path.exists(fn_response):
                sleep(0.5)

            with open(fn_response, 'rb') as handle:
                reply, label = pickle.load(handle)

            os.remove(fn_response)

            is_hastag = True if label in self.reply_hashtags else False

            self._log('Reply is ready {} | {} | {}'.format(reply, label, is_hastag))

        if custom_delay:
            word_count = len(reply.split())
            sleep(word_count // 2)

        return reply, label, is_hastag
