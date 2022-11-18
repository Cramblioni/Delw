
from ast import literal_eval
from collections import ChainMap
from typing import Callable, Any
from iters import *

__all__ = (
    "evaluate",
    "getPrelude",
    "getEnv"
)

class Sexpr(list):
    col: int
    line: int
    def __repr__(self): return f"$({', '.join(map(repr, self))})"

class Ident(str):
    def __repr__(self): return "£" + super().__str__()

def parse(txt: str) -> Sexpr:
    cur = Sexpr()
    result = Sexpr([cur])

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

    while cind < mxind:
        c = advance()
        if c in " \n\t": continue
        elif c == "/" and peek == "/":
            while peek != "\n": advance()
        elif c == "(":
            cur = Sexpr()
            cur.col = col
            cur.line = line
            result.append(cur)
        elif c == ")":
            result.pop()
            result[-1].append(cur)
            cur = result[-1]
        elif c == '"':
            buff: str = ""
            while peek != '"':
                if peek == "\\":
                    advance()
                    buff += advance()
                else: buff += advance()
            advance()
            cur.append(buff)
        elif c.isdigit():
            buff = c
            hd = False
            while peek.isdigit():
                buff += advance()
                if peek == "." and not hd:
                    hd = True; buff += advance()
            cur.append(literal_eval(buff))
        else:
            buff = c
            while peek not in "() \n\t":
                buff += advance()
            cur.append(Ident(buff))

    assert len(result) == 1
    return result[0][0]

def evaluate(path: str,
      env:Callable[[],dict[str, object]|ChainMap[str, object]]=dict) -> object:
    prog : Sexpr
    with open(path, "r") as f:
        prog = parse(f.read())
    venv = (lambda x: ChainMap(x) if isinstance(x, dict) else x)(env())
    venv = venv.new_child()
    try:
        return _eval(prog, venv)
    except Exception as e:
        raise Exception(f"near line {venv[' env'][' line']},"
                        f" col {venv[' env'][' col']} in {path}\n{e.args[0]}")\
            from e

class Macro:
    func: Callable[[ChainMap[str, object], Sexpr], Sexpr]
    def __init__(self, func): self.func = func

def _eval(prog: Sexpr | Ident | int | float | str, env: ChainMap[str, object]) -> object:
    if isinstance(prog, Ident): return env[str(prog)]
    if not isinstance(prog, Sexpr): return prog
    assert len(prog) > 0
    assert isinstance(env[" env"], ChainMap)
    env[" env"][" line"] = prog.line
    env[" env"][" col"] = prog.col
    if len(prog) == 1:
        if isinstance(prog[0], Sexpr):
            tmp = _eval(prog[0], env)
            return tmp() if callable(tmp) else tmp
        else:
            return _eval(prog[0], env)
    else:
        f = _eval(prog[0], env)
        if isinstance(f, Macro):
            return _eval(f.func(env, Sexpr(prog[1:])), env)
        elif callable(f):
            return f(*map(lambda x: _eval(x, env), prog[1:]))
        else: raise RuntimeError("cannot call non-callable")

class TagInterp:
    tag: str
    arr: list[Any]
    attrib: dict[str, str]
    __toTagOverride__: Callable[[], str]
    def __init__(self, *args):
        self.arr = list(args)
        self.attrib = {}
    def toTag(self):
        if hasattr(self, "__toTagOverride__"):
            return self.__toTagOverride__()
        attribs = map("%s=\"%s\"".__mod__, self.attrib.items())
        if len(self.arr) == 0:
            return f'<{" ".join(chain([self.tag], attribs))}/>'
        cont = "".join(map(digest, self.arr))
        return f'<{" ".join(chain([self.tag], attribs))}>{cont}</{self.tag}>'
    def update(self, dct: dict):
        self.attrib.update(dct)
        return self

def domgen(tags: list[str]) -> dict[str, object]:
    result = {}
    for tag in tags:
        result[tag] = type(tag, (TagInterp,),{"tag":tag[1:]})
    return result

tags = domgen([
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
    '_video', '_wbr'])
tags["_doctype"] = type("!DOCTYPE", (TagInterp,), {
    "tag": "!Doctype",
    "__toTagOverride__": lambda s: f"<!Doctype {s.arr[0]}>"
})

#@Macro
#def spread(func, itr):
#    return Sexpr([func, *itr])
def spread(func, itr):
    return func(*itr)

@Macro
def fIf(env, expr):
    (cond, iftrue, orelse) = expr
    if _eval(cond, env): return iftrue
    else: return orelse

@Macro
def fLet(env:ChainMap, expr):
    nenv = env.new_child(**{k: _eval(v, env)
                            for (k, v) in chunk(expr[:-1], 2)})
    return _eval(expr[-1], nenv)
@Macro
def fDef(env, expr):
    argnames = expr[:-1]
    val = expr[-1]
    def f(*args):
        nenv=env.new_child(**dict(zip(argnames, args)))
        return _eval(val, nenv)
    return f
@Macro
def include(env, expr):
    venv = ChainMap(env[" env"])
    return evaluate(str(_eval(expr[0], env)), lambda: venv)
@Macro
def do(env, expr):
    result = None
    for result in map(_eval, expr, repeat(env)):
        pass
    return result

class Morsel(str): pass
def digest(*args: Any) -> Morsel:
    if len(args) > 1:
        return Morsel("".join(map(digest, args)))
    [node] = args
    if hasattr(node, "toTag") and callable(node.toTag):
        return Morsel(node.toTag())
    if isinstance(node, Morsel): return node
    else:
        return Morsel(str(node).replace("<", "&lt;").replace(">", "&gt;"))

import operator
builtins = {
    "digest": digest, "include": include, "do": do,
    "getattr": getattr,
    "setattr": setattr,
    "getitem": lambda x, y: x[y],
    "setitem": lambda x, y, v: x.__setitem__(y, v),
    "...": spread, "if": fIf,
    "map": map, "filter": filter,
    "list": lambda *a: list(chain(*a)),
    "range": range,
    "let": fLet, "def": fDef,
    "format": str.format,
    "attrib": lambda x, *a: x.update(dict(chunk(a, 2))),
    ".": lambda f, g: lambda x: g(f(x)),
    "str":str, "int":int, "float":float,
    "empty": lambda x: x([]),
    **vars(operator),
    **tags
}
builtins[" env"] = builtins

def getPrelude(): return builtins
def getEnv(env: dict[str, object]) -> Callable[[], ChainMap[str, object]]:
    nenv = ChainMap(getPrelude(), env)
    nenv[" env"] = nenv
    return lambda: nenv

if __name__ == "__main__":
    test = "(digest (p \"this is a \" (b \"TEST\") \" boio\"))"
    data = parse(test)
    print(repr(data))

    print(_eval(data, ChainMap(builtins, {})))
    print(evaluate("./src/index.htmlisp", lambda: builtins))
