class NList:
    def __init__(self, lst, n=1):
        self.lst = lst
        self.n = n
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        l = self.lst[self.index:self.index+self.n:]
        self.index += self.n
        yield tuple((l + self.n * [None])[:self.n])