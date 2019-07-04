import requests
import json

URL = "http://127.0.0.1:8000/api_create_conversation/"
domain_id =  "1"
PARAMS = {'domain_id': domain_id}
r = requests.post(url=URL, data=PARAMS)
data = r.json()
print(data)
chat_id = data['data']['id']
# the repsonse is {u'message': None, u'data': {u'id': u'xxx'}, u'success': True}
# if the action is a successfull one on data you will receive the id of the record
lst_msg = []
lst_msg.append({'chat_id': chat_id,
                'human': False,
                'message': 'Buna User',
                'label': 'salut'})
lst_msg.append({'chat_id': chat_id,
                'human': True,
                'message': 'Buna Oana',
                'label': 'salut'})
lst_msg.append({'chat_id': chat_id,
                'human': False,
                'message': 'Cu ce te pot ajuta?',
                'label': 'question'})
lst_msg.append({'chat_id': chat_id,
                'human': True,
                'message': 'Ma doare rau capul',
                'label': 'sanatate'})

for msg in lst_msg:
    rm = requests.post(url="http://127.0.0.1:8000/api_create_message/", data=msg)

    dt = rm.json()
    print(dt)