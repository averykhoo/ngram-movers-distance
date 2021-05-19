import io
import string
import sys
import time
from numbers import Number
from typing import Callable
from typing import Generator
from typing import Mapping
from unittest.mock import sentinel

from pympler import asizeof

NOT_ITERABLE = (
    io.IOBase,
    str,
    bytes,
    bytearray,
    range,
    Number,
    Generator,
    Callable,
    type,
)


def deep_sizeof(obj):
    sizes = dict()
    stack = {id(obj): obj}

    def add_to_stack(thing):
        if id(thing) not in sizes:
            stack[id(thing)] = thing

    while stack:
        item_id, item = stack.popitem()

        # already counted
        if item_id in sizes:
            continue

        # special case for numpy objects
        size = getattr(item, 'nbytes', None)
        if isinstance(size, int):
            sizes[item_id] = size
            continue

        # special case for pandas dataframe and series
        if hasattr(item, 'memory_usage'):
            size = item.memory_usage(index=True, deep=True)
            if hasattr(size, 'sum'):
                size = size.sum()  # dataframes return a series which we need to sum
            if isinstance(size, int):
                sizes[item_id] = size
                continue

        # count size of item
        sizes[item_id] = sys.getsizeof(item)

        # nothing to recurse into
        if isinstance(item, NOT_ITERABLE):
            continue

        # recurse into dict-like item
        if isinstance(item, (Mapping, dict)):
            for key in item.keys():
                add_to_stack(key)
            for value in item.values():
                add_to_stack(value)

        # recurse into list-like item (but not range, range_iterator, map, filter, etc)
        # this will also recurse into a dict, but deduplication will handle this case
        if hasattr(item, '__iter__') and hasattr(item, '__getitem__'):
            for key in item:
                add_to_stack(key)

        # recurse into a class instance
        _dict = getattr(item, '__dict__', dict())
        for key in _dict.keys():
            add_to_stack(key)
        for value in _dict.values():
            add_to_stack(value)

        # recurse into a class instance (slots can co-exist with dict)
        _slots = getattr(item, '__slots__', ())
        for attr in _slots:
            add_to_stack(getattr(item, attr, None))

        # recurse into attributes (DANGER: still calls np.array().T recursively forever)
        for attr in dir(item):
            if not isinstance(getattr(type(item), attr, None), (property, type, Callable)):
                if attr[:2] != '__' and attr[-2:] != '__':  # can be fooled by magic method like names
                    add_to_stack(getattr(item, attr))

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
    depth = 10

    t = time.time()
    for _ in range(10):
        n1 = NodeA()
        head = n1
        for _ in range(depth):
            for char in charset:
                head[char] = NodeA()
                if char < 'M':
                    head[char].REPLACEMENT = char * 13
            head = head['a']
    t = time.time() - t
    print('n1', asizeof.asizeof(n1), deep_sizeof(n1), t)

    FLAG = object()
    t = time.time()
    for _ in range(10):
        n2 = dict()
        head = n2
        for _ in range(depth):
            for char in charset:
                head[char] = dict()
                if char < 'M':
                    head[char][FLAG] = char * 13
            head = head['a']
    t = time.time() - t
    print('n2', asizeof.asizeof(n2), deep_sizeof(n2), t)

    FLAG2 = object()
    t = time.time()
    for _ in range(10):
        n3 = dict()
        head = n3
        for _ in range(depth):
            for char in charset:
                head[char] = dict()
                if char < 'M':
                    head[char][FLAG] = FLAG2
            head = head['a']
    t = time.time() - t
    print('n3', asizeof.asizeof(n3), deep_sizeof(n3), t)

    flag = sentinel.flag
    t = time.time()
    for _ in range(10):
        n4 = dict()
        head = n4
        for _ in range(depth):
            for char in charset:
                head[char] = dict()
                if char < 'M':
                    head[char][flag] = char * 13
            head = head['a']
    t = time.time() - t
    print('n4', asizeof.asizeof(n4), deep_sizeof(n4), t)

    t = time.time()
    for _ in range(10):
        n5 = NodeB()
        head = n5
        for _ in range(depth):
            for char in charset:
                head[char] = NodeB()
                if char < 'M':
                    head[char].REPLACEMENT = char * 13
            head = head['a']
    t = time.time() - t
    print('n5', asizeof.asizeof(n5), deep_sizeof(n5), t)

    t = time.time()
    for _ in range(10):
        n6 = NodeC()
        head = n6
        for _ in range(depth):
            for char in charset:
                head.DATA[char] = NodeC()
                if char < 'M':
                    head.DATA[char].REPLACEMENT = char * 13
            head = head.DATA['a']
    t = time.time() - t
    print('n6', asizeof.asizeof(n6), deep_sizeof(n6), t)

    # print([(n, getattr(n1, n)) for n in dir(n1)])

    import numpy as np

    print(deep_sizeof(np.array(range(10))))
    print(deep_sizeof(np.array(range(1000))))

    import pandas as pd

    print(deep_sizeof(pd.DataFrame([[1, 2, 3], [4, 5, 6]])))
