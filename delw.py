import sys, os
import sqlite3
import hashlib

# http stuffs
import http.server
import http.cookies
import socketserver
from urllib.parse import urlparse, parse_qs


# hydration
import htmlisp

# handling sessions
import uuid
import datetime

from getpass import getpass

class Session:
  account: "int | None"
  def __init__(self):
    self.account = None

global DBCONN, SESSIONS
SESSIONS : dict[str, Session] = {}

PATH: str = "./delw.db"
SITEPATH: str = "./src/"
PERMIT_EDITALL = 1

def serialise(x: object):
  if type(x) is str: return f'"{repr(x)[1:-1]}"'
  else: return repr(x)

class APIthingy(http.server.BaseHTTPRequestHandler):

  def do_GET(self):
    print(">>>>>>>>>>>>> PATH:: ",self.path, file=sys.stderr)
    global DBCONN
    cookie = {}
    for i in self.headers.get("Cookie", "").split("; "):
      if not i: break
      (key, value) = i.split("=")
      cookie[key] = value

    tmp = urlparse(self.path)
    path, query = tmp.path, parse_qs(tmp.query)
    del tmp

    if path == "/":
      curs = DBCONN.cursor()
      venv = htmlisp.getEnv({
        "path": self.path,
        "cookie": cookie,
        "curs": curs,
        "session": SESSIONS.get(cookie["session"], None),
      })
      self.send_response(200)
      self.end_headers()
      self.wfile.write(
        str(htmlisp.evaluate("./src/index.htmlisp", venv)).encode()
      )
      curs.close()

    elif path.startswith("/app/"):
      print("APPLETTE")
      curs = DBCONN.cursor()
      venv = htmlisp.getEnv({
        "path": self.path,
        "cookie": cookie,
        "curs": curs,
        "session": SESSIONS.get(cookie["session"], None),
        "front": path[5:],
        "query": {k:v[0] for k, v in query.items()}
      })
      self.send_response(200)
      self.end_headers()
      self.wfile.write(
        str(htmlisp.evaluate("./src/app.htmlisp", venv)).encode()
      )
      curs.close()

    elif path.startswith("/src/"):
      filepath = SITEPATH + path[5:]
      if os.path.exists(filepath):
        self.send_response(200)
        self.end_headers()
        with open(filepath, "rb") as f:
          self.wfile.write(f.read())
      else:
        self.send_response(404)

    elif path.startswith("/images/"):
      imgpath = SITEPATH + path[1:]
      if os.path.exists(imgpath):
        self.send_response(200)

        ext =  path.rsplit(".", 1)[-1]
        if ext == "svg": ext = "svg+xml"
        self.send_header("content-type", "image/" + ext)
        self.end_headers()
        with open(imgpath, "rb") as f:
          self.wfile.write(f.read())
      else:
        self.send_response(404)
    else:
      self.send_error(404)

  def do_POST(self):
    if self.path == "/login":
      pass
    elif self.path == "/logout":
      pass
    else:
      self.send_error(404, "na, just na")

def createUser():
  pass

PORT: int = 8080
HANDLER = APIthingy

if __name__ == "__main__":
  if not os.path.exists(PATH):
    print("Cannot find delw.db")
    print("Creating database")
    conn = sqlite3.connect(PATH)
    curs = conn.cursor()
    print("Creating user table")
    curs.execute("""
CREATE TABLE "users" (
	"id"	INTEGER,
	"name"	TEXT,
	"password"	BLOB,
	PRIMARY KEY("id" AUTOINCREMENT)
)
""")
    print("creating posts table")
    curs.execute("""
CREATE TABLE "posts" (
	"id"	INTEGER,
	"poster"	INTEGER,
	"path"	TEXT,
	"title"	TEXT,
	"description"	TEXT,
	PRIMARY KEY("id" AUTOINCREMENT)
)
""")
    print("creating admin user")
    uname  = input(  "username: ")
    psswd  = getpass("password: ")
    phsh   = hashlib.sha256(bytes(psswd, "utf8")).digest()
    curs.execute("INSERT INTO users VALUES(0, ?, ?, ?)",
      (PERMIT_EDITALL, uname, phsh))
    print("Successfully created user", uname)

    curs.close()
    conn.commit()
    conn.close()
  print("Opening DB connection")
  DBCONN = sqlite3.connect(PATH)
  try:
    with socketserver.TCPServer(("", PORT), HANDLER) as httpd:
      print("serving at port", PORT)
      httpd.serve_forever()
  finally:
    print("Closing DB connection")
    DBCONN.close()
