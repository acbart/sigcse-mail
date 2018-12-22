import sqlite3
from bottle import Bottle, run, \
     template, debug, static_file, request
import os, sys
import json

app = Bottle()
dirname = ""

print("TEST")

@app.route('<filename:re:.*\.css>')
def send_css(filename):
    return static_file(filename, root='static/asset/css')

@app.route('<filename:re:.*\.js>')
def send_js(filename):
    return static_file(filename, root='static/asset/js')


@app.route('/')
def index():
    return template('index')
    
conn = sqlite3.connect('sigcse-emails.db')
c = conn.cursor()
c.execute("SELECT sent, sender, subject, body, kind, year, month, week FROM email e "
          "LEFT OUTER JOIN thread t ON t.id=e.thread_id "
          "WHERE kind = 'Normal' "
          "ORDER BY sent ASC "
          "LIMIT 10")
result = c.fetchall()
threads= [{
            'sent': r[0],
            'sender': r[1],
            'subject': r[2],
            'body': r[3],
            'kind': r[4],
            'year': r[5],
            'month': r[6],
            'week': r[7]
          } for r in result]

@app.route('/timeline')
def timeline():
    return template('timeline', rows=threads)

@app.route('/get_posts', method="GET")
def get_posts():
    searchTerm = request.GET.searchTerm.strip()
    
    conn = sqlite3.connect('sigcse-emails.db')
    c = conn.cursor()
    c.execute("SELECT sent, sender, subject, body, kind FROM email "
              "WHERE body LIKE ? "
              "AND kind = 'Normal'",
              ("%"+searchTerm+"%",))
    result = c.fetchall()
    return json.dumps([{
                'sent': r[0],
                'sender': r[1],
                'subject': r[2],
                'body': r[3],
                'kind': r[4]
            } for r in result])

debug(True)
run(app, host='localhost', port = 8080, reloader=True)