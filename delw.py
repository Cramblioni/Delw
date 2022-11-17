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

# tokens :)
import uuid

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
  global DBCONN, COOKIEIND
  def do_GET(self):
    cookie = {}
    for i in self.headers.get("Cookie", "").split("; "):
      if not i: break
      (key, value) = i.split("=")
      cookie[key] = value


    print("cookie session:", cookie.get("session", "not-set"))
    if ("Cookie" not in self.headers or cookie.get("session") not in SESSIONS)\
    and (self.path.startswith("/app/") or self.path == "/") :
      print("Handling cookies",
            "Cookie" not in self.headers,
            cookie.get("session") not in SESSIONS,
            SESSIONS)
      session = str(uuid.uuid1())
      SESSIONS[session] = Session()


      self.send_response(307)
      self.send_header("Location", self.path)
      cookie = http.cookies.SimpleCookie()
      cookie["session"] = session
      for morsel in cookie.values():
        self.send_header("Set-Cookie", morsel.OutputString())
      self.end_headers()
      return

    tmp = urlparse(self.path)
    path, query = tmp.path, parse_qs(tmp.query)
    del tmp

    if path == "/":
      curs = DBCONN.cursor()
      venv = htmlisp.getEnv({
        "path": self.path,
        "cookie": cookie,
        "curs": curs,
        "session": SESSIONS[cookie["session"]],
      })
      self.send_response(200)
      self.end_headers()
      self.wfile.write(
        str(htmlisp.evaluate("./src/index.htmlisp", venv)).encode()
      )
      curs.close()

    elif path.startswith("/image/"):
      imgpath = SITEPATH + path[1:]
      print(os.path.abspath(imgpath))
      if os.path.exists(imgpath):
        print("FOUND")
        self.send_response(200)
        self.send_header("content-type", "image/" + path.rsplit(".", 1)[-1])
        self.end_headers()
        with open(imgpath, "rb") as f:
          self.wfile.write(f.read())
      else:
        print("Not found")
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
	"id"	INTEGER UNIQUE,
	"permit" INTEGER,
	"name"	TEXT UNIQUE,
	"password"	BLOB,
	PRIMARY KEY("id" AUTOINCREMENT)
)
""")
    print("creating posts table")
    curs.execute("""
CREATE TABLE "posts" (
	"id"	INTEGER UNIQUE,
	"poster"	INTEGER,
	"path"	TEXT,
	"title"	TEXT,
	"description"	TEXT,
	"tags"	TEXT,
	FOREIGN KEY("poster") REFERENCES "users"("id"),
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
