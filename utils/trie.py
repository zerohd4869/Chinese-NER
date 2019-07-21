import collections


class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):

        current = self.root
        for letter in word:
            current = current.children[letter]
        current.is_word = True

    def search(self, word):
        current = self.root
        for letter in word:
            current = current.children.get(letter)

            if current is None:
                return False
        return current.is_word

    def startsWith(self, prefix):
        current = self.root
        for letter in prefix:
            current = current.children.get(letter)
            if current is None:
                return False
        return True

    def enumerateMatch(self, char, space="_", backward=False):
        matched = []
        # while len(word) > 1 does not keep character itself, while word keed character itself
        while len(char) > 1:
            if self.search(char):
                matched.append(space.join(char[:]))
            del char[-1]
        return matched
