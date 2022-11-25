import sys, os
from ast import literal_eval
from collections import ChainMap
from typing import Callable, Optional, \
                   Any, Iterable, NoReturn, TypeVar, Generic
from iters import *

global CURPATH
CURPATH : str

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
  __slots__ = ("toks", "end", "parent", "ind")
  toks: list[AstNode]
  end: int
  parent: "Expression | Program"
  ind: int
  def __init__(self, toks: list[AstNode], line: int, col: int, end: int,
                     parent: "Expression | Program", ind: int = -1):
    self.toks, self.line, self.col, self.end = toks, line, col, end
    self.parent, self.ind = parent, ind
  def __str__(self): return f"({' '.join(map(str, self.toks))})"
  def rci(self): return len(self.toks)

class Program(AstNode):
  __slots__  = ("prog", "path")
  prog: Optional[AstNode]
  path: str
  def __init__(self, prog: Optional[AstNode], path: str, line: int, col: int):
    self.prog, self.line, self.col = prog, line, col
    self.path = path
  def __str__(self): return str(self.prog)
  def rci(self): return -1

def getSelf(node: Expression | Program) -> Optional[AstNode]:
  """
  Gets itself from its parent.
  used for when a macro could change the structure of the AST
  """
  if isinstance(node, Program): return None
  if isinstance(node.parent, Program): return node.parent.prog
  return node.parent.toks[node.ind]

def setSelf(node: Expression, new: AstNode):
  parent = node.parent
  if isinstance(parent, Program): parent.prog = new
  else: parent.toks[node.ind] = new


import pickle
def deepCopy(ast: AstNode):
  return pickle.loads(pickle.dumps(ast))

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
      push(Expression([], line, col, -1, stack[-1],stack[-1].rci()))
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
      while cind < mxind and peek not in "() \n\t":
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
  print("parsing:: ", repr(path))
  if not os.path.exists(path):
    return Error(f"file {path!r} does not exist", -1, -1, -1, path)
  with open(path, "rb") as f:
    text = f.read().decode("utf8")
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
        # creating the replacement
      else:
        stack.extend(cur.toks)
    else:
      continue

  return result

# evaluating this stuff
Environment = tuple[dict[str, Any], ChainMap[str, Any]]
from typing import Literal as Static

T = TypeVar("T")
T0 = TypeVar("T0");T1 = TypeVar("T1")
class Result(Generic[T]):
  __slots__ = ("kind", "val", "err")
  kind: bool
  err: Error
  val: T

  def __str__(self):
    if self.kind:
      return f"Error <{self.err}>"
    else:
      return f"Success <{self.val}>"

  def fmap(self, f: Callable[[T], T0]) -> "Result[T0]":
    if self.kind:
      return Failure(self.err)
    else:
      return Success(f(self.val))

def Failure(err: Error) -> Result[Any]:
  if not isinstance(err, Error):
    raise TypeError
  result = Result()
  result.kind = True
  result.err = err
  return result

def Success(val: T) -> Result[T]:
  if isinstance(val, Result):
    raise TypeError
  result = Result()
  result.kind = False
  result.val = val
  return result

class Macro:
  func: Callable[[Expression, Environment, Iterable[AstNode]], Optional[Error]]
  def __init__(self, func): self.func = func

def _eval(node: AstNode, env: Environment) -> Result[Any]:
  if type(node) is Program:
    global CURPATH
    assert node.prog is not None
    nenv = (env[0], ChainMap({"__name__": node.path}))
    CURPATH = node.path
    return _eval(deepCopy(node.prog), nenv)
  elif type(node) is Expression:
    # if and while the first one is a macro, expand it, hard re-evaluate
    # else treat as function call
    fnode = _eval(node.toks[0], env)
    if fnode.kind: return fnode

    if isinstance(fnode.val, Macro):
      err_ = fnode.val.func(node, env, node.toks[1:][:])
      if err_ is not None: return Failure(err_)
      targ = getSelf(node)
      if targ is None:
        return Failure(Error("Macro expansion created empty expression",
                      node.line, node.col, node.end,
                      env[1]["__name__"]))
      return _eval(targ, env)
    args = []
    for an in node.toks[1:]:
      arg = _eval(an, env)
      if arg.kind: return arg
      args.append(arg.val)
    if isinstance(fnode, Error): return Failure(fnode)
    return fnode.val(*args)

  elif type(node) is Ident:
    res: Any; err: Optional[Error]
    try: return Success(env[node.tok in env[1]][node.tok])
    except KeyError: return Failure(Error(
      f"name {node.tok!r} not bound",
      node.line, node.col, -1,
      env[1].get("__name__", "__main__")
    ))
  elif type(node) is Literal:
    return Success(literal_eval(node.lit))

  elif type(node) is Raw:
    return node.val

  return Failure(Error("Unreachable... maybe not?",
               node.line, node.col, -1,
               env[1].get("__name__", "__main__")))

__all__ = (
    "evaluate",
    "getPrelude",
    "getEnv",
    "Result", "showError", "Error",
    "Success", "Failure"
)

###############################################################

def ResultWrap(f: Callable[..., Any]) -> Callable[..., Result]:
  global CURPATH
  def wrapped(*args):
    try:
      return Success(f(*args))
    except Exception as e:
      return Failure(Error(str(e), -1, -1, -1 , CURPATH))
  return wrapped
# implementing HTML tags

class HtmlTag:
  __slots__ = ("attrib", "children")
  tag: str
  attrib: dict[str, str]
  children: list["HtmlTag | Any"]

  def __init__(self, *children):
    self.children = list(children)
    self.attrib = {}
  def _html_(self) -> str:
    intag = " ".join(chain([self.tag],
                           map('%s="%s"'.__mod__, self.attrib.items())))
    if len(self.children) == 0:
      return f"<{intag} />"
    return f'<{intag}>{"".join(map(html, self.children))}</{self.tag}>'

def html(obj: Any) -> str:
  if hasattr(obj, "_html_"):
    return obj._html_()
  elif isinstance(obj, str):
    return obj.replace(">", "&gt;").replace("<", "&lt;")
  raise TypeError

def tagGen(tags: list[str]) -> dict[str, HtmlTag]:
  result = {}
  for tag in tags:
    result[tag] = ResultWrap(type(tag, (HtmlTag,), {"tag": tag[1:]}))
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
    "_doctype": ResultWrap(type("_doc", (HtmlTag,), {
      "_html_": lambda s:f"<!DOCTYPE {s.children[0]}>"
    }))}

class Morsel(str):
  def _html_(self): return str(self)
def digest(*tags: HtmlTag) -> Morsel:
  return Morsel("".join(map(html, tags)))

def attrib(tag: HtmlTag, *attribs: str) -> HtmlTag:
  tmp = type(tag)()
  tmp.children = tag.children[:]
  tmp.attrib = {**tag.attrib, **dict(chunk(attribs, 2))}
  return tmp

@Macro
def fIf(ref: Expression, env: Environment, args: list[AstNode]) -> Optional[Error]:
  if len(args) != 3:
    return Error("`if` takes 3 arguments",
                 ref.line, ref.col, ref.end, env[1]["__name__"])
  cond = _eval(args[0], env)
  if cond.kind: return cond.err
  setSelf(ref, args[2 - bool(cond.val)])


@Macro
def fLet(ref: Expression, env: Environment, args: list[AstNode]) -> Optional[Error]:
  if len(args) < 3:
    return Error("let requires atleast 3 args",
                 ref.line, ref.col, ref.end, env[1]["__name__"])
  if len(args) & 1 == 0:
    return Error("let should name value pairs followed by an expression",
                 ref.line, ref.col, ref.end, env[1]["__name__"])
  expr = args[-1]
  nenv = {}
  for (k, v) in chunk(args[:-1], 2):
    v2 = _eval(v, env)
    if v2.kind:
      return v2.err
    nenv[k.tok] = v2.val
  setSelf(ref, Raw(_eval(expr, (env[0], env[1].new_child(nenv))),
                         ref.line, ref.col))

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
    params.append(pn.tok)
  def inner(*args) -> Result:
    if len(args) != len(params):
      return Failure(Error(f"defun expected {len(params)} args, got {len(args)}",
                    ref.line, ref.col, ref.end, env[1]["__name__"]))
    nenv = dict(zip(params, args))
    return _eval(body, (env[0], env[1].new_child(nenv)))

  setSelf(ref, Raw(Success(inner), ref.line, ref.col))

@Macro
def fEnv(ref: Expression, env: Environment, args: list[AstNode]) -> Optional[Error]:
  if len(args) == 0:
    return Error("Expected atleast 1 arg for `¦`",
                 ref.line, ref.col, ref.end, env[1]["__name___"])
  if len(args) == 1:
    # this is get
    lookup = _eval(args[0], env).fmap(str)
    if lookup.kind: return lookup.err
    try:
      script = env[0]["__scripts__"][lookup.val]
    except KeyError:
      return Error(f"Script \"{lookup.val}\" not loaded",
                           ref.line, ref.col, ref.end,
                           env[1]["__name__"])
    setSelf(ref, Raw(_eval(script, env), ref.line, ref.col))

def spread(f:Callable[..., Any], args: Iterable[Any]) -> Any:
  return f(*args)
def wMap(f: Callable[..., Result], xs: Iterable[Any]) -> Result:
  result = []
  for x in xs:
    res = f(x)
    if res.kind: return res
    result.append(res.val)
  return Success(result)
_benv = {
    "+": ResultWrap(lambda x, *y: reduce(op.add, y, x)),
    "-": ResultWrap(lambda x, *y: reduce(op.sub, y, x)),
    "*": ResultWrap(lambda x, *y: reduce(op.mul, y, x)),
    "/": ResultWrap(lambda x, *y: reduce(op.truediv, y, x)),
    "//": ResultWrap(lambda x, *y: reduce(op.floordiv, y, x)),
    "%": ResultWrap(lambda x, *y: reduce(op.mod, y, x)),
    "=": ResultWrap(lambda x, *y: reduce(op.eq, y, x)),
    "/=": ResultWrap(lambda x, *y: reduce(op.ne, y, x)),
    "...": ResultWrap(lambda f, xs: f(*xs)),
    "[]": ResultWrap(list),
    "range": ResultWrap(range),
    "map": wMap,
    "list": ResultWrap(lambda *x: list(x)),
    "digest": ResultWrap(digest), "attrib": ResultWrap(attrib),

    "." : ResultWrap(getattr),
    ".=": ResultWrap(setattr),

    "if": fIf, "let": fLet, "def": fDef,
    "include": fEnv,

    **tags
  }

def getPrelude():
  return lambda :(_benv, ChainMap())
def evaluate(path: str,
             env: Callable[[], Environment]) -> Result:
  prog = parseFile(path)
  if isinstance(prog, Error):
    showError(prog)
    return Failure(prog)
  ltable = link(prog, {"path": prog})
  if isinstance(ltable, Error):
    showError(ltable)
    return Failure(ltable)
  venv = env()
  venv = ({**venv[0], "__scripts__": ltable}, venv[1])
  return _eval(prog, venv)


def getEnv(env: dict[str, object]):
  return lambda :(_benv, ChainMap(env))

def showError(err: Error):
  print("---- ERERT")
  lines = []
  print(err, file= sys.stderr)
  with open(err.file, "r") as f:
    for _ in range(err.line - 1):
      f.readline()
    lines.append(f.readline())
    if err.end != -1:
      for _ in range(err.end - err.line):
        lines.append(f.readline())

  print(*lines, sep="\n",file=sys.stderr)
  if len(lines) == 1:
    print(" "*(err.col - 2) + "^", file=sys.stderr)

from functools import reduce
import operator as op

if __name__ == "__main__":
  filename = "./test.htmlisp"
  resp = evaluate(filename, getPrelude())
  if resp.kind:
    showError(resp.err)
    exit(1)
  print(resp.val)
