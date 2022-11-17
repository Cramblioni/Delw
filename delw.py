import sys, os
import sqlite3
import hashlib

from getpass import getpass

class Image:
  path: str
  title: str
  description: str
  tags: list[str]

def main(dump):
  pass

PATH: str = ".\\delw.db"
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
	"name"	TEXT,
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
    curs.execute("INSERT INTO users VALUES(0, ?, ?)", (uname, phsh))
    print("Successfully created user", uname)

    curs.close()
    conn.commit()
    conn.close()  
