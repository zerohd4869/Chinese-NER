import collections


class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False


class Trie:
    def __init__(self, use_single):
        self.root = TrieNode()
        if use_single:
            self.min_len = 0
        else:
            self.min_len = 1

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

    def enumerateMatch(self, word, space="_", backward=False):
        matched = []
        # while len(word) > 1 does not keep character itself, while word keed character itself
        while len(word) > self.min_len:
            if self.search(word):
                matched.append(space.join(word[:]))
            del word[-1]
        return matched
