from bokeh.io import curdoc
from chatbot import Server
from time import time

start = time()
server = Server(bot_name="Allan",
                config_file='config_server.txt',
                show_bot_messages=True)

layouts = server.CreateLayout()

for layout in layouts:
    curdoc().add_root(layout)

doc = curdoc()
args = doc.session_context.request.arguments
doc.add_periodic_callback(callback=server._standard_end_message,
                          period_milliseconds=360000)

curdoc().title = "Temprent"
