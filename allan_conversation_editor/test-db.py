import time
start = time.time()
import MySQLdb

db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                     user="root",         # your username
                     passwd="iuli",  # your password
                     db="allan")        # name of the data base

# you must create a Cursor object. It will let
#  you execute all the queries you need
cur = db.cursor()

# Use all the SQL you like
cur.execute("SELECT * FROM `conversation_editor_chat`")

# print all the first cell of all the rows
for row in cur.fetchall():
    row#print(row[0])

db.close()
end = time.time()
print("Time to get all records from chats table in MYSQL")
print(end - start)

start = time.time()
import pyodbc
conn = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=lummetry-allan.database.windows.net;DATABASE=allan-conversations;UID=lummetry;PWD=MLteam2019!')

cursor = conn.cursor()
cursor.execute('SELECT * FROM conversation_editor_chat')

for row in cursor:
    row#print(row)
end = time.time()
print("Time to get all records from chats table in MSSQL")
print(end - start)