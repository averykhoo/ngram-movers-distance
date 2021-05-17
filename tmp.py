import io
import string
import sys
from typing import Callable
from typing import Generator
from typing import Mapping
from unittest.mock import sentinel

from pympler import asizeof

NON_LIST_ITERABLES = (
    io.IOBase,
    str,
    bytes,
    bytearray,
    range,
)


def deep_sizeof(obj):
    sizes = dict()
    stack = [obj]

    while stack:
        item = stack.pop(-1)

        # already counted
        if id(item) in sizes:
            continue

        # count size of item
        sizes[id(item)] = sys.getsizeof(item)

        # recurse into dict-like item
        # noinspection PyTypeChecker
        if isinstance(item, (Mapping, dict)):
            stack.extend(item.keys())
            stack.extend(item.values())

        # recurse into list-like item
        elif hasattr(item, '__iter__') and hasattr(item, '__getitem__') and not isinstance(item, NON_LIST_ITERABLES):
            stack.extend(item)

        # recurse into a class instance
        if hasattr(item, '__dict__'):
            stack.extend(item.__dict__.keys())
            stack.extend(item.__dict__.values())

        # get any additional variables
        else:
            for attr in dir(item):
                if not (attr.startswith('__') or attr.endswith('__')):
                    if not isinstance(getattr(item, attr), (Callable, type, Generator)):
                        stack.append(getattr(item, attr))

    return sum(sizes.values())


_NOTHING = object()


class NodeA(dict):
    __slots__ = ('REPLACEMENT',)

    # noinspection PyMissingConstructor
    def __init__(self):
        self.REPLACEMENT = _NOTHING


class NodeB(dict):
    __slots__ = ()

    @property
    def REPLACEMENT(self):
        return self.get(_NOTHING, _NOTHING)

    @REPLACEMENT.setter
    def REPLACEMENT(self, value):
        if value == _NOTHING:
            del self[_NOTHING]
        else:
            self[_NOTHING] = value


class NodeC:
    __slots__ = ('DATA', 'REPLACEMENT')

    # noinspection PyMissingConstructor
    def __init__(self):
        self.DATA = dict()
        self.REPLACEMENT = _NOTHING

    # def __getitem__(self, item):
    #     return self.DATA[item]

    # def __setitem__(self, key, value):
    #     self.DATA[key] = value


if __name__ == '__main__':

    charset = string.printable
    depth = 20

    n1 = NodeA()
    head = n1
    for _ in range(depth):
        for char in charset:
            head[char] = NodeA()
            if char < 'M':
                head[char].REPLACEMENT = char * 13
        head = head['a']
    print('n1', asizeof.asizeof(n1), deep_sizeof(n1))

    FLAG = object()
    n2 = dict()
    head = n2
    for _ in range(depth):
        for char in charset:
            head[char] = dict()
            if char < 'M':
                head[char][FLAG] = char * 13
        head = head['a']
    print('n2', asizeof.asizeof(n2), deep_sizeof(n2))

    FLAG2 = object()
    n3 = dict()
    head = n3
    for _ in range(depth):
        for char in charset:
            head[char] = dict()
            if char < 'M':
                head[char][FLAG] = FLAG2
        head = head['a']
    print('n3', asizeof.asizeof(n3), deep_sizeof(n3))

    flag = sentinel.flag
    n4 = dict()
    head = n4
    for _ in range(depth):
        for char in charset:
            head[char] = dict()
            if char < 'M':
                head[char][flag] = char * 13
        head = head['a']
    print('n4', asizeof.asizeof(n4), deep_sizeof(n4))

    n5 = NodeB()
    head = n5
    for _ in range(depth):
        for char in charset:
            head[char] = NodeB()
            if char < 'M':
                head[char].REPLACEMENT = char * 13
        head = head['a']
    print('n5', asizeof.asizeof(n5), deep_sizeof(n5))

    n6 = NodeC()
    head = n6
    for _ in range(depth):
        for char in charset:
            head.DATA[char] = NodeC()
            if char < 'M':
                head.DATA[char].REPLACEMENT = char * 13
        head = head.DATA['a']
    print('n6', asizeof.asizeof(n6), deep_sizeof(n6))

    print([(n, getattr(n1, n)) for n in dir(n1)])
