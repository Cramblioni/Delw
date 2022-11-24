import sys, os, asyncio
import sqlite3
import hashlib
from operator import eq, ne
from functools import partial
# typehints because QOL
from typing import Optional
# http stuffs
import http.server
import http.cookies
import socketserver
from urllib.parse import urlparse, parse_qs

# hydration
import htmlisp
from collections import ChainMap

# handling sessions
import uuid
import datetime

from getpass import getpass

class Session:
  __slots__ = ("id","permit", "name", "created", "sid")
  id: int
  permit: int
  name: str
  created: str
  sid: str
  def __init__(self: "Session", id: int, permit: int,
               name: str, created: str, sid: str):
    self.id, self.permit, self.sid = id, permit, sid
    self.name, self.created = name, created


global DBCONN, SESSIONS, CURLOOP
SESSIONS : dict[str, Session] = {}

DATEFORMAT : str = "%Y%H%M%S"

PATH: str = "./delw.db"
SITEPATH: str = "./src/"
PERMIT_EDITALL = 1
PERMIT_POST    = 2
PERMIT_EDIT    = 4

def genSessionMeta():
  return str(uuid.uuid4()).replace("-", " ") ,\
         datetime.datetime.now().strftime(DATEFORMAT)

class APIthingy(http.server.BaseHTTPRequestHandler):

  def do_GET(self):
    print(">>>>>>>>>>>>> PATH:: ",self.path, file=sys.stderr)
    global DBCONN
    cookie = parseHeader(self.headers.get("Cookie", ""))

    tmp = urlparse(self.path)
    path, query = tmp.path, parse_qs(tmp.query)
    del tmp

    if path == "/":
      curs = DBCONN.cursor()
      venv = lambda :({
        "path": self.path,
        "cookie": cookie,
        "curs": curs,
        "session": SESSIONS.get(cookie["session"], None)
                   if "session" in cookie else None,
        "front": path[5:],
        "method": "POST",
        "query": {k:v[0] for k, v in query.items()},
        **htmlisp.getPrelude()()[0]
      },  ChainMap({}))
      err, msg = htmlisp.evaluate("./src/index.htmlisp", venv)
      if err is not None:
        htmlisp.showError(err)
        self.send_response(501)
        return
      self.send_response(200)
      self.end_headers()
      self.wfile.write(
        str(msg).encode()
      )
      curs.close()

    elif path.startswith("/app/"):
      print("APPLETTE")
      curs = DBCONN.cursor()
      venv = lambda :({
        "path": self.path,
        "cookie": cookie,
        "curs": curs,
        "session": SESSIONS.get(cookie["session"], None)
                   if "session" in cookie else None,
        "front": path[5:],
        "method": "GET",
        "query": {k:v[0] for k, v in query.items()},
        **htmlisp.getPrelude()()[0]
      },  ChainMap({}))
      err, msg = htmlisp.evaluate("./src/app.htmlisp", venv)
      if err is not None:
        htmlisp.showError(err)
        self.send_response(501)
        return
      self.send_response(200)
      self.end_headers()
      self.wfile.write(
        str(msg).encode()
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
    global DBCONN
    cookie = parseHeader(self.headers.get("Cookie", ""))

    tmp = urlparse(self.path)
    path, query = tmp.path, parse_qs(tmp.query)
    del tmp
    UNV_0 : dict[str, Item] = {
      parseHeader(i.header["Content-Disposition"])["name"] : i
      for i in parsePOST(self)}

    # setup then do API stuff
    if path == "/app/login":
      pass
    elif path == "/app/upload":
      pass
    elif path == "/app/signup":
      pass

    if path.startswith("/app/"):
      curs = DBCONN.cursor()
      venv = lambda :({
        "path": self.path,
        "cookie": cookie,
        "curs": curs,
        "session": SESSIONS.get(cookie["session"], None)
                   if "session" in cookie else None,
        "front": path[5:],
        "method": "POST",
        "query": {k:v[0] for k, v in query.items()},
        **htmlisp.getPrelude()()[0]
      },  ChainMap({}))
      self.send_response(200)
      self.end_headers()
      self.wfile.write(
        str(htmlisp.evaluate("./src/app.htmlisp", venv)).encode()
      )
      curs.close()
    else:
      self.send_error(404, "na, just na")

def login(uname: str, psswd: str) ->Session|str:
  global DBCONN
  curs = DBCONN.cursor()
  curs.execute("""
  SELECT id, permit, name, password FROM users WHERE name = ?
  """, (uname,))

  res = curs.fetchone()
  if res is None: return "Cannot find user name {uname!r}"

  uid, upermit, _, upsswd = res

  psswdHsh = hashpass(psswd)
  if not (len(psswdHsh) == len(upsswd) and all(map(eq, upsswd, psswdHsh))):
    return "Username or Password Incorrect"

  curs.close()

  return Session(uid, upermit, uname, *genSessionMeta())

def hashpass(psswd: str) -> bytes:
  return hashlib.sha256(psswd.encode()).digest()

def createUser(curs: sqlite3.Cursor, name: str,
               psswd: str, permit: int):
  phsh   = hashlib.sha256(bytes(psswd, "utf8")).digest()
  curs.execute("INSERT INTO users (permit, name, password) VALUES(?, ?, ?)",
    (permit, name, phsh))

def createPost():
  pass

def parseHeader(header: str) -> dict[str, str]:
  return dict(map(lambda x: x.split("="),
              filter(partial(ne, ""), header.split("; "))))

class Item:
  header: dict[str, str]
  data: bytes

  def __init__(self, header, data):
    self.header,self.data=header,data

def parsePOST(self: APIthingy) -> list[Item]:
  """
  Reads the body of the POST request and various headers and returns
  all extracted information.
  """
  # getting the payload
  body = self.rfile.read(int(self.headers["Content-Length"]))
  # some more header parsing
  contentType, *contentMeta = self.headers["Content-Type"].split("; ")
  contentMeta = dict(map(lambda x:x.split("="), contentMeta))
  if contentType == "multipart/form-data":
    divider = b"\r\n--" + contentMeta["boundary"]
    # annoying
    result = []
    for chunk in body.split(divider):
      # check the first 2 bytes, "\r\n" means heres a chunk
      #                          "--"   means end of multipart
      if chunk[:2] == b'--': break
      assert chunk[:2] == b'\r\n'
      header, payload = chunk[2:].split(b"\r\n\r\n", 1)
      header = {k.decode():v.decode()
                for k, v in
                map(lambda x: x.split(b":",1), header.split(b"\r\n"))}
      result.append(Item(header, payload))
    return result
  elif contentType == "application/x-www-form-urlencoded":
    return [Item({
      k.decode(): v
      for (k, v) in map(lambda x: x.split(b"="), body.split(b"&"))}, b"")]
  return []

PORT: int = 8080
HANDLER = APIthingy
async def main():
  global DBCONN, CURLOOP
  CURLOOP = asyncio.get_running_loop()
  if not os.path.exists(PATH):
    print("Cannot find delw.db")
    print("Creating database")
    conn = sqlite3.connect(PATH)
    curs = conn.cursor()
    print("Creating user table")
    curs.execute("""
CREATE TABLE "users" (
	"id"	INTEGER,
  "permit"	INTEGER,
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
  "tags" TEXT,
	PRIMARY KEY("id" AUTOINCREMENT)
)
""")
    print("creating admin user")
    uname  = input(  "username: ")
    psswd  = getpass("password: ")
    createUser(curs, uname, psswd, PERMIT_EDITALL | PERMIT_POST)
    print("Successfully created user", uname)

    curs.close()
    conn.commit()
    conn.close()
  print("Opening DB connection")
  DBCONN = sqlite3.connect(PATH)
  try:
    with socketserver.TCPServer(
      server_address=("", PORT),
      RequestHandlerClass=HANDLER)\
    as httpd:
      print("serving at port", PORT)
      httpd.serve_forever()
  finally:
    print("Closing DB connection")
    DBCONN.close()

if "__main__" == __name__:
  asyncio.run(main=main())
