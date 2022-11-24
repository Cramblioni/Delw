import sys, os
from ast import literal_eval
from collections import ChainMap
from typing import Callable, Optional, Any
from iters import *


class Error:
  __slots__ = ("msg", "line", "col", "file", "end")
  msg: str
  line: int
  col: int
  endl: int
  file: str

  def __init__(self, msg: str, line: int, col: int, end: int, file: str):
    self.msg, self.line, self.col = msg, line, col
    self.file, self.end = file, end

  def __str__(self):
    return f"{self.file}:{self.line}:{self.col}: {self.msg}"

class AstNode:
  __slots__ = ("line", "col")
  line: int
  col: int

  def __str__(self): return "<Node>"

class Ident(AstNode):
  __slots__ = ("tok",)
  tok: str
  def __init__(self, tok: str, line: int, col: int):
    self.tok, self.line, self.col = tok, line, col
  def __str__(self): return f"*{self.tok}"

class Literal(AstNode):
  __slots__ = ("lit",)
  lit: str
  def __init__(self, lit: str, line: int, col: int):
    self.lit, self.line, self.col = lit, line, col
  def __str__(self): return f"'{self.lit}"

class Raw(AstNode):
  __slots__ = ("val",)
  val: Any
  def __init__(self, val: Any, line: int, col: int):
    self.val, self.line, self.col = val, line, col

class NoCall(AstNode):
  __slots__ = ("val",)
  val: Any
  def __init__(self, val: Any, line: int, col: int):
    self.val, self.line, self.col = val, line, col

class Expression(AstNode):
  __slots__ = ("toks", "end")
  toks: list[AstNode]
  end: int
  def __init__(self, toks: list[AstNode], line: int, col: int, end: int):
    self.toks, self.line, self.col, self.end = toks, line, col, end
  def __str__(self): return f"({' '.join(map(str, self.toks))})"


class Program(AstNode):
  __slots__  = ("prog", "path")
  prog: Optional[AstNode]
  path: str
  def __init__(self, prog: Optional[AstNode], path: str, line: int, col: int):
    self.prog, self.line, self.col = prog, line, col
    self.path = path
  def __str__(self): return str(self.prog)


def parse(txt: str, path="undefined") -> Program | Error:
  cind, mxind = 0, len(txt)
  line, col = 1, 1
  peek: str = txt[0] if mxind > 0 else "\x00"
  def advance() -> str:
    nonlocal peek, cind, line, col
    cind, ret = cind + 1, peek
    col += 1
    if ret == "\n": col, line = 1, line + 1
    peek = txt[cind] if cind < mxind else "\x00"
    return ret
  stack : list[Expression] = []
  def push(x: Expression): stack.append(x)
  def pop() -> Expression:
    result = stack.pop()
    assert isinstance(result, Expression)
    return result
  def genErr(msg, endl=-1):
    return Error(msg, line, col, endl, path)

  while cind < mxind:
      if peek in " \n\t": advance()
      elif (peek == "/" and peek == "/") or peek == ";":
        while peek != "\n": advance()
      else:
        break
  else:
      return genErr("No Program Found")
  def add(x: AstNode):
      if isinstance(stack[-1], Program):
        setattr(stack[-1], "prog", x)
      else:
        stack[-1].toks.append(x)
  list.append(stack, Program(None, path, line, col))
  while cind < mxind:
    c = advance()
    if c in " \n\t": continue
    elif (c == "/" and peek == "/") or c == ";":
      while peek != "\n": advance()
    elif c == "(":
      push(Expression([], line, col, -1))
    elif c == ")":
      tmp00: Expression = pop()
      tmp00.end = line
      if len(stack) == 0:
        return genErr("unpaired Closin paren")
      add(tmp00)
    elif c == '"':
      buff: str = c
      while peek != '"':
        if peek == "\\":
          advance()
          buff += advance()
        else: buff += advance()
      buff += advance()
      add(Literal(buff, line, col))
    elif c.isdigit():
      buff = c
      hd = False
      if peek == "." and not hd:
        hd = True; buff += advance()
      while peek.isdigit():
        buff += advance()
        if peek == "." and not hd:
          hd = True; buff += advance()
      add(Literal(buff, line, col))
    else:
      buff = c
      while peek not in "() \n\t":
        buff += advance()
      if buff == "None": add(Literal(buff, line, col))
      else: add(Ident(buff, line, col))
    if isinstance(stack[-1], Program):
      tmp : Program = list.pop(stack)
      while cind < mxind:
        if peek == ";":
          while advance() != "\n": pass
        elif advance() not in " \n\t": break
      else:
        return tmp
      return genErr("garbage after program")
  return genErr("This is not suppose to happen")

def quickCheck(arg: Program, path="undefined") -> Optional[Error]:
  assert arg.prog is not None
  def genError(tok: AstNode, msg, endl=-1):
    return Error(msg, tok.line, tok.col, endl, path)

  # check for empty expressions or expr.toks == []
  # And Here's an example of obverver recursion
  stack : list[AstNode] = [arg.prog]
  while len(stack) > 0:
    cur = stack.pop()
    if isinstance(cur, Expression):
      if len(cur.toks) == 0:
        return genError(cur, "Empty expression located")
      else:
        stack.extend(cur.toks)
    else:
      continue
  # we should do a second traversal to find other
  # errors that are easily known (spreading nothing, let with no bindings)
  return None

def parseFile(path: str) -> Program | Error:
  if not os.path.exists(path):
    return Error(f"file {path!r} does not exist", -1, -1, -1, path)
  with open(path, "r") as f:
    text = f.read()
  step = parse(text, path)
  if isinstance(step, Error): return step
  test = quickCheck(step, path)
  if isinstance(test, Error): return test
  return step

def link(arg: Program, lookup: dict[str, Program]) -> dict[str, Program] | Error:
  assert arg.prog is not None
  def genError(tok: AstNode, msg, endl=-1):
    return Error(msg, tok.line, tok.col, endl, arg.path)

  result = {**lookup}
  stack : list[AstNode] = [arg.prog]
  while len(stack) > 0:
    cur = stack.pop()
    if isinstance(cur, Expression):
      if isinstance(cur.toks[0], Ident) and cur.toks[0].tok == "include":
        if len(cur.toks) != 2:
          return genError(cur, "`inlcude` must have one argument, which is "\
                               "a static string")
        a, pathtok = cur.toks
        if not isinstance(pathtok, Literal):
          return genError(pathtok, "`inlcude` argument must be static string")
        path = literal_eval(pathtok.lit)
        if not isinstance(path, str):
          return genError(pathtok, "`inlcude` argument must be string")
        if path in result: continue
        if not os.path.exists(path):
          return genError(pathtok, f"Cannot Find source file {path!r}")
        iprog = parseFile(path)
        if isinstance(iprog, Error): return iprog
        result[path] = iprog
        tmp = link(iprog, result)
        if isinstance(tmp, Error): return tmp
        result.update(tmp)
        cur.toks = [NoCall(result[path], a.line, a.col)]
      else:
        stack.extend(cur.toks)
    else:
      continue

  return result

# evaluating this stuff
Environment = tuple[dict[str, Any], ChainMap[str, Any]]
Result = tuple[Optional[Error], Any]

class Macro:
  func: Callable[[Expression, Environment, tuple[AstNode, ...]], Result]
  def __init__(self, func): self.func = func

def _eval(node: AstNode, env: Environment) -> Result:
  if type(node) is Program:
    assert node.prog is not None
    nenv = (env[0], ChainMap({"__name__": node.path}))
    return _eval(node.prog, nenv)
  elif type(node) is Expression:
    # check if dynmacro then change how shit is passed
    fnode = node.toks[0]
    if isinstance(fnode, NoCall):
      return _eval(fnode.val, env)
    err, f  = _eval(fnode, env)
    if err is not None: return err, None
    if isinstance(f, Macro):
      err, new = f.func(node, env, tuple(node.toks[1:]))
      if err is not None: return err, None
      if not isinstance(new, list):
        new= [new]
      return _eval(Expression(list(new), node.line, node.col, node.end), env)
    else:
      if not callable(f):
        return Error("attempted call of non-callable",
                     node.line, node.col, node.end,
                     env[1]["__name__"]),None
      args = []
      for i in node.toks[1:]:
        err, val  = _eval(i, env)
        if err is not None: return err, None
        args.append(val)

      return None, f(*args)

  elif type(node) is Ident:
    res: Any; err: Optional[Error]
    try: err, res = None, env[node.tok in env[1]][node.tok]
    except KeyError: res, err = None, Error(
      f"name {node.tok!r} not bound",
      node.line, node.col, -1,
      env[1].get("__name__", "__main__")
    )
    return err, res
  elif type(node) is Literal:
    return None, literal_eval(node.lit)

  elif type(node) is Raw:
    return None, node.val

  return Error("Unreachable... maybe not?",
               node.line, node.col, -1,
               env[1].get("__name__", "__main__")), None

__all__ = (
    "evaluate",
    "getPrelude",
    "getEnv",
    "Result", "showError", "Error",
)

# implementing HTML tags

class HtmlTag:
  __slots__ = ("attrib", "children")
  tag: str
  attrib: dict[str, str]
  children: list["HtmlTag | Any"]

  def __init__(self, *children):
    self.children = list(children)
    self.attrib = {}
  def __str__(self) -> str:
    intag = " ".join(chain([self.tag],
                           map('%s="%s"'.__mod__, self.attrib.items())))
    if len(self.children) == 0:
      return f"<{intag} />"
    return f'<{intag}>{"".join(map(str, self.children))}</{self.tag}>'

def tagGen(tags: list[str]) -> dict[str, HtmlTag]:
  result = {}
  for tag in tags:
    result[tag] = type(tag, (HtmlTag,), {"tag": tag[1:]})
  return result

def _dts(self): return f"<!DOCTYPE {self.children[0]}"
tags = {**tagGen([
    '_a', '_abbr', '_address', '_area', '_article', '_aside', '_audio', '_b',
    '_base', '_bdi', '_bdo', '_blockquote', '_body', '_br', '_button',
    '_canvas', '_caption', '_cite', '_code', '_col', '_colgroup', '_data',
    '_datalist', '_dd', '_del', '_details', '_dfn', '_dialog', '_div', '_dl',
    '_dt', '_em', '_embed', '_fieldset', '_figcaption', '_figure', '_footer',
    '_form', '_h1', '_h2', '_h3', '_h4', '_h5',  'h6', '_head', '_header',
    '_hr', '_html', '_i', '_iframe', '_img', '_input', '_ins', '_kbd',
    '_label', '_legend', '_li', '_link', '_main', '_map', '_mark', '_meta',
    '_meter', '_nav', '_noscript', '_object', '_ol', '_optgroup', '_option',
    '_output', '_p', '_param', '_picture', '_pre', '_progress', '_q', '_rp',
    '_rt', '_ruby', '_s', '_samp', '_script', '_section', '_select', '_small',
    '_source', '_span', '_strong', '_style', '_sub', '_summary', '_sup',
    '_svg', '_table', '_tbody', '_td', '_template', '_textarea', '_tfoot',
    '_th', '_thead', '_time', '_title', '_tr', '_track', '_u', '_ul', '_var',
    '_video', '_wbr']),
    "_doctype": type("_doc", (HtmlTag,), {"tag": "!DOCTYPE", "__str__": _dts})}

class Morsel(str): pass
def digest(*tags: HtmlTag) -> Morsel:
  return Morsel("".join(map(str, tags)))

def attrib(tag: HtmlTag, *attribs: str) -> HtmlTag:
  tmp = type(tag)()
  tmp.children = tag.children[:]
  tmp.attrib = {**tag.attrib, **dict(chunk(attribs, 2))}
  return tmp

@Macro
def fIf(ref: Expression, env: Environment, args: list[AstNode]) -> Result:
  if len(args) != 3:
    return Error("`if` takes 3 arguments",
                 ref.line, ref.col, ref.end, env[1]["__name__"]), None
  err, cond = _eval(args[0], env)
  if err is not None: return err, None
  body = args[int(cond) + 1]
  return None, body.toks if isinstance(body, Expression) else [body]

@Macro
def fLet(ref: Expression, env: Environment, args: list[AstNode]):
  if len(args) < 3:
    return Error("let requires atleast 3 args",
                 ref.line, ref.col, ref.end, env[1]["__name__"]), None
  if len(args) & 1 == 0:
    return Error("let should name value pairs followed by an expression",
                 ref.line, ref.col, ref.end, env[1]["__name__"]), None
  expr = args[-1]
  nenv = {}
  for (k, v) in chunk(args[:-1], 2):
    err, v2 = _eval(v, env)
    if err is not None:
      return err, None
    nenv.update(((k, v),))

  err, ret = _eval(expr, (env[0], env[1].new_child(nenv)))
  if err is not None:
    return err, None
  return None, Raw(ret, ref.line, ref.col)

@Macro
def fDef(ref: Expression, env: Environment, args: list[AstNode]):
  if len(args) < 1:
    return Error("`def` requires body",
                 ref.line, ref.col, ref.end, env[1]["__name__"]), None
  body = args[-1]
  params = []
  for pn in args[:-1]:
    if not isinstance(pn, Ident):
      return Error("`def` requires body",
                  ref.line, ref.col, ref.end, env[1]["__name__"]), None
    params.append(pn)
  def inner(*iargs : Any):
    nenv = {k.toks:v for (k,v) in zip(params, iargs)}
    return _eval(body, (env[0], env[1].new_child(nenv)))

  return None, Raw(inner, ref.line, ref.col)


_benv = {
    "+": lambda x, *y: reduce(op.add, y, x),
    "-": lambda x, *y: reduce(op.sub, y, x),
    "*": lambda x, *y: reduce(op.mul, y, x),
    "/": lambda x, *y: reduce(op.truediv, y, x),
    "//": lambda x, *y: reduce(op.floordiv, y, x),
    "%": lambda x, *y: reduce(op.mod, y, x),
    "=": lambda x, *y: reduce(op.eq, y, x),
    "/=": lambda x, *y: reduce(op.ne, y, x),
    "...": lambda f, xs: f(*xs),
    "[]": list,
    "range": range,
    "map": map,
    "list": lambda *x: list(x),
    "digest": digest, "attrib": attrib,

    "if": fIf, "let": fLet, "def": fDef,

    **tags
  }

def getPrelude():
  return lambda :(_benv, ChainMap())
def evaluate(path: str,
             env: Callable[[], Environment]) -> Result:
  prog = parseFile(path)
  if isinstance(prog, Error):
    showError(prog)
    return prog, None
  ltable = link(prog, {"path": prog})
  if isinstance(ltable, Error):
    showError(ltable)
    return ltable, None
  return _eval(prog, env())


def getEnv(env: dict[str, object]):
  return lambda :(_benv, ChainMap(env))

def showError(err: Error):
  lines = []
  with open(err.file, "r") as f:
    for _ in range(err.line - 1):
      f.readline()
    lines.append(f.readline())
    if err.end != -1:
      for _ in range(err.end - err.line):
        lines.append(f.readline())
  print(err, file= sys.stderr)
  print(*lines, sep="\n",file=sys.stderr)
  if len(lines) == 1:
    print(" "*(err.col - 2) + "^", file=sys.stderr)

from functools import reduce
import operator as op

if __name__ == "__main__":
  filename = "./src/index.htmlisp"
  resp = parseFile(filename)
  if isinstance(resp, Error):
    showError(resp)
    exit(1)
  ltable = link(resp, {filename: resp})
  if isinstance(ltable, Error):
    showError(ltable)
    exit(1)
  err, ret = _eval(resp, (_benv, ChainMap({})))
  if err is not None:
    showError(err)
    exit(1)
  print(ret)
