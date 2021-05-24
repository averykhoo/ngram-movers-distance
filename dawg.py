"""
see: https://gist.github.com/smhanov/94230b422c2100ae4218

"""
import io
import random
import time


class DAWGNode(object):
    """
    This class represents a node in the directed acyclic word graph (DAWG).
    It has a list of edges to other nodes.
    It has functions for testing whether it is equivalent to another node.
    Nodes are equivalent if they have identical edges, and each identical edge leads to identical states.
    The __hash__ and __eq__ functions allow it to be used as a key in a python dictionary.
    """
    NEXT_ID = 0

    def __init__(self):
        # set unique ID
        self.id = DAWGNode.NEXT_ID
        DAWGNode.NEXT_ID += 1

        # node data
        self.is_terminal_node = 0
        self.edges = {}

        # finalized data
        self.final_leaf_count = None
        self.final_edges = None

    def finalize(self):
        if self.final_leaf_count is None:
            # init
            self.final_leaf_count = 0

            # count the number of final nodes that are reachable from this, including self
            if self.is_terminal_node:
                self.final_leaf_count += 1
            for label, node in sorted(self.edges.items()):
                node.finalize()
                self.final_leaf_count += node.final_leaf_count

            # sort the edges for faster lookup
            self.final_edges = sorted(self.edges.items())

    def __str__(self):
        edge_info = u''

        for label, node in sorted(self.edges.items()):
            edge_info += u' {LABEL}: {ID},'.format(LABEL=repr(label), ID=node.id)

        return u'DAWGNode(terminal={T}, edges={{{E}}}'.format(T=self.is_terminal_node, E=edge_info[1:-1])

    def __hash__(self):
        return self.__str__().__hash__()

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __iter__(self):
        if self.is_terminal_node:
            yield u''
        for child_label, child_node in sorted(self.edges.items()):
            for char_seq in child_node:
                yield child_label + char_seq

    def __getitem__(self, index):
        index -= self.is_terminal_node
        if index == -1:
            return u''
        for label, node in self.final_edges:
            if index < node.final_leaf_count:
                return label + node[index]
            index -= node.final_leaf_count

    def clone(self):
        assert self.final_leaf_count is None
        assert self.final_edges is None

        new_node = DAWGNode()
        new_node.is_terminal_node = self.is_terminal_node
        for label, node in self.edges.items():
            new_node.edges[label] = node

        return new_node


class DAWG(object):
    """
    make a DAWG
    """

    def __init__(self):
        self.previousWord = u''
        self.root = DAWGNode()

        # Here is a list of nodes that have NOT been checked for duplication.
        self.unchecked_nodes = []

        # Here is a list of unique nodes that have been checked for duplication.
        self.minimized_nodes = {}

        # Here is the data associated with all the nodes
        self.data = []

    def insert(self, new_word, data=None):
        # sanity checks
        new_word = new_word
        assert new_word > self.previousWord, \
            'ERROR: Insertions must be in alphabetical order: "{P}" --> "{C}"'.format(P=self.previousWord, C=new_word)

        # find common prefix between word and previous word
        common_prefix_len = 0
        for i, j in zip(new_word, self.previousWord):
            if i == j:
                common_prefix_len += 1
            else:
                break

        # Check the uncheckedNodes for redundant nodes, proceeding from last one down to the common prefix size.
        # Then truncate the list at that point; remaining un-minimized nodes are the common prefix.
        self.minimize_unchecked(common_prefix_len)

        # add data
        self.data.append(data)

        # get the starting node
        if self.unchecked_nodes:
            node = self.unchecked_nodes[-1][-1]
        else:
            node = self.root

        # add the rest of the word (without the common prefix)
        for letter in new_word[common_prefix_len:]:
            next_node = DAWGNode()
            node.edges[letter] = next_node
            self.unchecked_nodes.append((node, letter, next_node))
            node = next_node

        # mark terminal node
        node.is_terminal_node = 1

        # remember last word
        self.previousWord = new_word

    def finalize(self):
        # minimize all uncheckedNodes
        self.minimize_unchecked(0)

        # go through entire structure and assign the counts to each node.
        self.root.finalize()

    def minimize_unchecked(self, prefix_len):
        # proceed from the leaf up to a certain point
        for _ in range(prefix_len, len(self.unchecked_nodes)):
            parent, letter, child = self.unchecked_nodes.pop()
            # if duplicate node exists
            if str(child) in self.minimized_nodes:
                # replace the child with the previously encountered one
                parent.edges[letter] = self.minimized_nodes[str(child)]
                del child
            else:
                # add the state to the minimized nodes.
                self.minimized_nodes[str(child)] = child

    def index(self, word):
        current_node = self.root
        skipped_leaf_count = 0
        partial_path = [skipped_leaf_count, current_node]
        for char in word:
            partial_path = self.step(partial_path, char)
            if partial_path is None:
                return -1
        skipped_leaf_count, current_node = partial_path
        if current_node.is_terminal_node:
            return skipped_leaf_count
        return -1

    def lookup(self, word):
        index = self.index(word)
        if index >= 0:
            return self.data[index]

    @staticmethod
    def step(partial_path, char):
        for child_label, child_node in partial_path[1].final_edges:
            if char == child_label:
                partial_path[0] += partial_path[1].is_terminal_node
                partial_path[1] = child_node
                break
            partial_path[0] += child_node.final_leaf_count
        else:
            return None  # char not found
        return partial_path

    def count_nodes(self):
        return len(self.minimized_nodes)

    def count_edges(self):
        count = 0
        for node in self.minimized_nodes.values():
            count += len(node.final_edges)
        return count

    def __iter__(self):
        for word in self.root:
            yield word

    def __len__(self):
        return len(self.data)

    def __contains__(self, item):
        return self.index(item) >= 0

    def keys(self):
        return list(self.__iter__())

    def items(self):
        return zip(self.__iter__(), self.data)

    def values(self):
        return self.data

    def get_index(self, i):
        assert i < len(self.data)
        return self.root[i]


class DAWG2(object):
    """
    make a DAWG not in alphabetical order
    """

    def __init__(self):
        self.previousWord = u''
        self.root = DAWGNode()

        # Here is a list of nodes that have NOT been checked for duplication.
        self.unchecked_nodes = []

        # Here is a list of unique nodes that have been checked for duplication.
        self.minimized_nodes = {}

        # Here is the data associated with all the nodes
        self.data = []

    def insert(self, new_word, data=None):
        # sanity checks
        new_word = new_word
        assert new_word > self.previousWord, \
            'ERROR: Insertions must be in alphabetical order: "{P}" --> "{C}"'.format(P=self.previousWord, C=new_word)

        # find common prefix between word and previous word
        common_prefix_len = 0
        for i, j in zip(new_word, self.previousWord):
            if i == j:
                common_prefix_len += 1
            else:
                break

        # Check the uncheckedNodes for redundant nodes, proceeding from last one down to the common prefix size.
        # Then truncataverye the list at that point; remaining un-minimized nodes are the common prefix.
        self.minimize_unchecked(common_prefix_len)

        # add data
        self.data.append(data)

        # get the starting node
        if self.unchecked_nodes:
            node = self.unchecked_nodes[-1][-1]
        else:
            node = self.root

        # add the rest of the word (without the common prefix)
        for letter in new_word[common_prefix_len:]:
            next_node = DAWGNode()
            node.edges[letter] = next_node
            self.unchecked_nodes.append((node, letter, next_node))
            node = next_node

        # mark terminal node
        node.is_terminal_node = 1

        # remember last word
        self.previousWord = new_word

    def finalize(self):
        # minimize all uncheckedNodes
        self.minimize_unchecked(0)

        # go through entire structure and assign the counts to each node.
        self.root.finalize()

    def minimize_unchecked(self, prefix_len):
        # proceed from the leaf up to a certain point
        for _ in range(prefix_len, len(self.unchecked_nodes)):
            parent, letter, child = self.unchecked_nodes.pop()
            # if duplicate node exists
            if str(child) in self.minimized_nodes:
                # replace the child with the previously encountered one
                parent.edges[letter] = self.minimized_nodes[str(child)]
                del child
            else:
                # add the state to the minimized nodes.
                self.minimized_nodes[str(child)] = child

    def index(self, word):
        current_node = self.root
        skipped_leaf_count = 0
        partial_path = [skipped_leaf_count, current_node]
        for char in word:
            partial_path = self.step(partial_path, char)
            if partial_path is None:
                return -1
        skipped_leaf_count, current_node = partial_path
        if current_node.is_terminal_node:
            return skipped_leaf_count
        return -1

    def lookup(self, word):
        index = self.index(word)
        if index >= 0:
            return self.data[index]

    @staticmethod
    def step(partial_path, char):
        for child_label, child_node in partial_path[1].final_edges:
            if char == child_label:
                partial_path[0] += partial_path[1].is_terminal_node
                partial_path[1] = child_node
                break
            partial_path[0] += child_node.final_leaf_count
        else:
            return None  # char not found
        return partial_path

    def count_nodes(self):
        return len(self.minimized_nodes)

    def count_edges(self):
        count = 0
        for node in self.minimized_nodes.values():
            count += len(node.final_edges)
        return count

    def __iter__(self):
        for word in self.root:
            yield word

    def __len__(self):
        return len(self.data)

    def __contains__(self, item):
        return self.index(item) >= 0

    def keys(self):
        return list(self.__iter__())

    def items(self):
        return zip(self.__iter__(), self.data)

    def values(self):
        return self.data

    def get_index(self, i):
        assert i < len(self.data)
        return self.root[i]


def yield_lines(file_path, make_lower=False, threshold_len=0):
    """
    yields all non-empty lines in a file
    :param file_path: file to read
    :param make_lower: force line to lowercase
    :param threshold_len: ignore lines equal <= this length
    """
    for line in io.open(file_path, mode='r', encoding='utf-8'):
        line = line.strip()
        if make_lower:
            line = line.lower()
        if len(line) > threshold_len:
            yield line


if __name__ == '__main__':

    dawg = DAWG()
    WordCount = 0
    # words = sorted(set(map(str.strip, list(open('buzhash.py')))))
    # words = []
    words = list(yield_lines('words.txt'))
    for num in range(100000000, 100100000):
        word_to_insert = ''
        for char in str(num):
            while random.random() < 0.01:
                word_to_insert += random.choice('+++-//////////////             000\\')
            word_to_insert += char
        words.append(word_to_insert)
    random.shuffle(words)
    print(words[:100])
    print('building...')
    start = time.time()
    for word in sorted(set(words)):
        if not word:
            continue
        WordCount += 1
        dawg.insert(word, word[::-1])

    dawg.finalize()
    print("dawg creation took {0} s".format(time.time() - start))

    EdgeCount = dawg.count_edges()
    print("Read {0} words into {1} nodes and {2} edges".format(
        WordCount, dawg.count_nodes(), EdgeCount))

    print("This could be stored in as little as {0} bytes".format(EdgeCount * 4))

    for word in ['test', 'noodle', 'fax', 'malware']:
        result = dawg.lookup(word)
        if result == None:
            print("{0} is NOT in dictionary.".format(word))
        else:
            print("{0} is in the dictionary and has data {1}".format(word, result))

    indices = list(range(len(dawg)))
    random.shuffle(indices)
    t = time.time()
    for _ in range(100000):
        index = indices.pop()
        word = dawg.get_index(index)
        assert dawg.index(word) == index

    print(time.time() - t)
