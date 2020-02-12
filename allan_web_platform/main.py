from bokeh.io import curdoc
from chatbot import Server
from time import time

start = time()
server = Server(bot_name="ALLAN", show_bot_messages=True,
                config_file='config_runner.txt')

layouts = server.CreateLayout()

for layout in layouts:
    curdoc().add_root(layout)

doc = curdoc()
args = doc.session_context.request.arguments
doc.add_periodic_callback(callback=server._standard_end_message,
                          period_milliseconds=360000)

curdoc().title = "Temprent"
print('FINAL TIME: {}'.format(time() - start))
