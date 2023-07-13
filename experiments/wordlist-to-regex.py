from find_replace_trie import Trie

if __name__ == '__main__':
    # example from https://github.com/ermanh/trieregex
    t1 = Trie.fromkeys(['understand', 'understate', 'undertake', 'undergo'])
    print(t1.to_regex())
    t2 = Trie.fromkeys(['grapefruit', 'grape', 'tangerine', 'tangelo', 'kumquat'])
    print(t2.to_regex())

    t3 = Trie.fromkeys([
        'a',
        'aardvark',
        'abaci',
        'aback',
        'abacus',
        'abacuses',
        'abandon',
        'abandoned',
        'abandoning',
        'abandonment',
        'abandons',
        'abate',
        'abated',
        'abates',
        'abating',
        'abbey',
        'abbeys',
        'abbot',
        'abbots',
        'abbreviate',
    ], case_sensitive=False, sort=True)
    print(t3.to_regex())
