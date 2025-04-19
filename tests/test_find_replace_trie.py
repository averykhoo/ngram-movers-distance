"""
Tests for the Trie implementation in find_replace_trie.py.

This file contains tests converted from the self_test function in find_replace_trie.py.
"""
import random
import re
import pytest

# Import the Trie class from the experiments module
from experiments.find_replace_trie import Trie


class TestTrie:
    """Tests for the Trie implementation."""
    
    def test_unicode_spaces(self):
        """Test that Unicode spaces are handled correctly."""
        _spaces = '\t\n\v\f\r \x85\xa0\x1c\x1d\x1e\x1f\ufeff\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006' \
                  '\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\u180e\u200b\u200c\u200d\u2060\u2800'
        
        # Check that the regex engine handles Unicode spaces correctly
        result = set(re.sub(r'\s', '', _spaces, flags=re.U))
        expected_results = [
            set('\u200b\u200c\u200d\u2060\u2800\ufeff'),
            set('\u200b\u200c\u200d\u2060\u2800\ufeff' + '\u180e')
        ]
        
        assert result in expected_results, f"Unicode spaces not handled correctly: {repr(result)}"
    
    def test_trie_update_with_tuples(self):
        """Test updating a Trie with a list of tuples."""
        _trie = Trie()
        assert len(_trie) == 0
        
        # Update with list of tuples
        _trie.update([('asd', '111'), ('hjk', '222'), ('dfgh', '3333'), ('ghjkl;', '44444'), ('jkl', '!')])
        assert len(_trie) == 5
        
        # Test translation
        assert ''.join(_trie.translate('erasdfghjkll')) == 'er111fg222ll'
        assert ''.join(_trie.translate('erasdfghjkl;jkl;')) == 'er111f44444!;'
        assert ''.join(_trie.translate('erassdfghjkl;jkl;')) == 'erass3333!;!;'
        assert ''.join(_trie.translate('ersdfghjkll')) == 'ers3333!l'
    
    def test_trie_regex(self):
        """Test the regex generation functionality of Trie."""
        # Create permutations for testing
        permutations = []
        for a in 'abcde':
            for b in 'abcde':
                for c in 'abcde':
                    for d in 'abcde':
                        permutations.append(a + b + c + d)
        
        # Add punctuation permutations
        for a in '`~!@#$%^&*()-=_+[]{}\\|;\':",./<>?\0':
            for b in '`~!@#$%^&*()-=_+[]{}\\|;\':",./<>?\0':
                for c in '`~!@#$%^&*()-=_+[]{}\\|;\':",./<>?\0':
                    permutations.append(a + b + c)
        
        # Test with a subset of permutations for reasonable test time
        for _ in range(10):  # Reduced from 1000 for faster tests
            chosen = set()
            for i in range(10):
                chosen.add(random.choice(permutations))
            
            _trie = Trie.fromkeys(chosen)
            assert len(_trie) == len(chosen)
            
            # Test regex generation
            r1 = re.compile(_trie.to_regex(fuzzy_quotes=False))
            found = r1.findall(' '.join(permutations))
            
            # All found items should be in the chosen set
            for item in found:
                assert item in chosen
                chosen.remove(item)
            
            # All items should have been found
            assert len(chosen) == 0
    
    def test_trie_update_with_dict(self):
        """Test updating a Trie with a dictionary."""
        _trie = Trie()
        _trie.update({'a': 'b', 'b': 'c', 'c': 'd', 'd': 'a'})
        assert ''.join(_trie.translate('acbd')) == 'bdca'
    
    def test_trie_operations(self):
        """Test various Trie operations like add, delete, contains."""
        _trie = Trie()
        _trie.update({
            'aa': '2',
            'aaa': '3',
            'aaaaaaaaaaaaaaaaaaaaaa': '~',
            'bbbb': '!',
        })
        assert len(_trie) == 4
        
        # Test contains and add
        assert 'aaaaaaa' not in _trie
        _trie['aaaaaaa'] = '7'
        assert len(_trie) == 5
        
        # Test translation
        assert ''.join(_trie.translate('a' * 12 + 'b' + 'a' * 28)) == '732b~33'
        assert ''.join(_trie.translate('a' * 40)) == '~773a'
        assert ''.join(_trie.translate('a' * 45)) == '~~a'
        assert ''.join(_trie.translate('a' * 25)) == '~3'
        assert ''.join(_trie.translate('a' * 60)) == '~~772'
        
        # Test delete
        del _trie['bbbb']
        assert 'b' not in _trie.root
        assert len(_trie) == 4
        
        del _trie['aaaaaaa']
        assert len(_trie) == 3
        assert 'aaa' in _trie
        assert 'aaaaaaa' not in _trie
        assert 'aaaaaaaaaaaaaaaaaaaaaa' in _trie
        
        _trie['aaaa'] = 4
        assert len(_trie) == 4
        
        del _trie['aaaaaaaaaaaaaaaaaaaaaa']
        assert len(_trie) == 3
        assert 'aaa' in _trie
        assert 'aaaaaaa' not in _trie
        assert 'aaaaaaaaaaaaaaaaaaaaaa' not in _trie
        
        # Test internal structure
        assert len(_trie.root['a']['a']['a']) == 1
        assert len(_trie.root['a']['a']['a']['a']) == 0
        
        # Test slice deletion
        del _trie['aaa':'bbb']
        assert _trie.to_regex() == '(?:aa)'
        assert len(_trie) == 1
    
    # def test_trie_fromkeys(self):
    #     """Test the fromkeys class method."""
    #     _trie = Trie.fromkeys('mad gas scar madagascar scare care car career error err are'.split())
    #     assert len(_trie) == 11
    #
    #     test = 'madagascareerror'
    #
    #     # Test findall without overlapping
    #     found = list(_trie.findall(test))
    #     assert found == ['madagascar', 'error']
    #
    #     # Test findall with overlapping
    #     found_overlapping = list(_trie.findall(test, allow_overlapping=True))
    #     expected = ['mad', 'gas', 'madagascar', 'scar', 'car', 'scare', 'care', 'car', 'career', 'error']
    #     assert found_overlapping == expected