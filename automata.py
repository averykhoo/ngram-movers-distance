"""
from 'Damn Cool Algorithms' blog
"""

import bisect
import time


class NFA(object):
    """
    non-deterministic finite state automaton
    from some state, given a transition, can end up at multiple states
    """
    EPSILON = object()
    ANY = object()

    def __init__(self, start_state):
        self.transitions = {}
        self.final_states = set()
        self._start_state = start_state

    @property
    def start_state(self):
        return frozenset(self._expand({self._start_state}))

    def add_transition(self, src, input, dest):
        self.transitions.setdefault(src, {}).setdefault(input, set()).add(dest)

    def add_final_state(self, state):
        self.final_states.add(state)

    def is_final(self, states):
        return self.final_states.intersection(states)

    def _expand(self, states):
        """
        expands a set of states
        to include states that are only epsilon-transitions away
        """
        frontier = set(states)
        while frontier:
            state = frontier.pop()
            new_states = self.transitions.get(state, {}).get(NFA.EPSILON, set()).difference(states)
            frontier.update(new_states)
            states.update(new_states)
        return states

    def next_state(self, states, input):
        dest_states = set()
        for state in states:
            state_transitions = self.transitions.get(state, {})
            dest_states.update(state_transitions.get(input, []))
            dest_states.update(state_transitions.get(NFA.ANY, []))
        return frozenset(self._expand(dest_states))

    def get_inputs(self, states):
        """
        outgoing transitions
        """
        inputs = set()
        for state in states:
            inputs.update(self.transitions.get(state, {}).keys())
        return inputs

    def to_dfa(self):
        dfa = DFA(self.start_state)
        frontier = [self.start_state]
        seen = set()
        while frontier:
            current = frontier.pop()
            inputs = self.get_inputs(current)
            for input in inputs:
                if input == NFA.EPSILON: continue
                new_state = self.next_state(current, input)
                if new_state not in seen:
                    frontier.append(new_state)
                    seen.add(new_state)
                    if self.is_final(new_state):
                        dfa.add_final_state(new_state)
                if input == NFA.ANY:
                    dfa.set_default_transition(current, new_state)
                else:
                    dfa.add_transition(current, input, new_state)
        return dfa


class DFA(object):
    """
    deterministic finite state automaton
    from some state, given a transition, goes to a single next state
    """

    def __init__(self, start_state):
        self.start_state = start_state
        self.transitions = {}
        self.defaults = {}
        self.final_states = set()

    def add_transition(self, src, input, dest):
        self.transitions.setdefault(src, {})[input] = dest

    def set_default_transition(self, src, dest):
        self.defaults[src] = dest

    def add_final_state(self, state):
        self.final_states.add(state)

    def is_final(self, state):
        return state in self.final_states

    def next_state(self, src, input):
        state_transitions = self.transitions.get(src, {})
        return state_transitions.get(input, self.defaults.get(src, None))

    def next_valid_string(self, input):
        stack = []

        # Evaluate the DFA as far as possible
        state = self.start_state
        for i, x in enumerate(input):
            stack.append((input[:i], state, x))
            state = self.next_state(state, x)
            if not state: break
        else:
            stack.append((input[:i + 1], state, None))

        # Input word is already valid?
        if self.is_final(state):
            return input

        # Perform a 'wall following' search for the lexicographically smallest accepting state.
        while stack:
            path, state, x = stack.pop()
            x = self.find_next_edge(state, x)
            if x is not None:
                path += x
                state = self.next_state(state, x)
                if self.is_final(state):
                    return path
                stack.append((path, state, None))
        return None

    def find_next_edge(self, state, transition):
        next_allowed_transition = u'\0' if transition is None else chr(ord(transition) + 1)
        state_transitions = self.transitions.get(state, {})
        if next_allowed_transition in state_transitions or state in self.defaults:
            return next_allowed_transition
        labels = sorted(state_transitions.keys())
        pos = bisect.bisect_left(labels, next_allowed_transition)
        if pos < len(labels):
            return labels[pos]
        return None


def levenshtein_automaton(word, k):
    nfa = NFA((0, 0))
    for index, char in enumerate(word):
        for dist in range(k + 1):
            # Correct character
            nfa.add_transition((index, dist), char, (index + 1, dist))  # edit here to make it case insensitive
            if dist < k:
                # Deletion
                nfa.add_transition((index, dist), NFA.ANY, (index, dist + 1))
                # Insertion
                nfa.add_transition((index, dist), NFA.EPSILON, (index + 1, dist + 1))
                # Substitution
                nfa.add_transition((index, dist), NFA.ANY, (index + 1, dist + 1))
    for dist in range(k + 1):
        if dist < k:
            nfa.add_transition((len(word), dist), NFA.ANY, (len(word), dist + 1))
        nfa.add_final_state((len(word), dist))
    return nfa


def find_all_matches(word, k, lookup_func):
    """Uses lookup_func to find all words within levenshtein distance k of word.

    Args:
      word: The word to look up
      k: Maximum edit distance
      lookup_func: A single argument function that returns the first word in the
        database that is greater than or equal to the input argument.
    Yields:
      Every matching word within levenshtein distance k from the database.
    """
    lev = levenshtein_automaton(word, k).to_dfa()
    match = lev.next_valid_string(u'\0')
    while match:
        next = lookup_func(match)
        if not next:
            return
        if match == next:
            yield match
            next += u'\0'
        match = lev.next_valid_string(next)


class Matcher(object):
    def __init__(self, entries):
        # self.sorted_entries = sorted(entries)
        self.sorted_entries = entries
        self.probes = 0

    def __call__(self, word):
        self.probes += 1
        pos = bisect.bisect_left(self.sorted_entries, word)
        if pos < len(self.sorted_entries):
            return self.sorted_entries[pos]
        else:
            return None


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if not s1:
        return len(s2)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            # j+1 instead of j since previous_row and current_row are one character longer than s2
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


class BKNode(object):
    def __init__(self, term):
        self.term = term
        self.children = {}
        self.results = []

    def insert(self, other):
        distance = levenshtein(self.term, other)
        if distance in self.children:
            self.children[distance].insert(other)
        else:
            self.children[distance] = BKNode(other)

    def search(self, term, k, results=None):
        if results is None:
            results = []
        distance = levenshtein(self.term, term)
        counter = 1
        if distance <= k:
            results.append(self.term)
        for i in range(max(0, distance - k), distance + k + 1):
            child = self.children.get(i)
            if child:
                counter += child.search(term, k, results)
        self.results = results
        return counter


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    words = sorted(line.split(',')[0].strip().lower() for line in open('british-english-insane.txt'))
    # words = sorted(line.split(',')[0].strip().lower() for line in open('words_en.txt'))
    print(len(words))

    bkn = BKNode('banana')
    for word in sorted(words):
        bkn.insert(word)

    xs = range(10)
    ts = []  # times
    ps = []  # probes
    fs = []  # found
    for i in xs:
        m = Matcher(words)
        t = time.time()
        print('-' * 100)
        found = list(find_all_matches('asalamalaikum', i, m))
        # found = list(find_all_matches('bananananaan', i, m))
        # found = list(find_all_matches('noodles', i, m))
        print('distance:', i)
        ts.append(time.time() - t)
        print('time:', time.time() - t)
        ps.append(float(m.probes) / len(words))
        print('probes:', m.probes, '=', float(m.probes) / len(words))
        fs.append(len(found))
        print('found:', len(found), found[:25])

        t = time.time()
        print(bkn.search('asalamalaikum', k=i))
        print(len(bkn.results), sorted(bkn.results)[:25])
        print(time.time()-t)

    plt.twinx().plot(xs, ts, '-r', label='time')
    plt.twinx().plot(xs, ps, '-g', label='probes')
    plt.twinx().plot(xs, fs, '-b', label='found')
    plt.show()
