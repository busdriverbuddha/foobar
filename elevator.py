
from functools import total_ordering

@total_ordering
class Version:

    def __init__(self, version_text):

        self.version = list(map(int, version_text.split(".")))
    

    def __eq__(self, other):

        if len(self.version) != len(other.version):
            return False

        for n1, n2 in zip(self.version, other.version):
            if n1 != n2:
                return False

        return True


    def __lt__(self, other):
        sv = self.version[:]
        ov = other.version[:]

        while sv and ov:
            s, o = sv.pop(0), ov.pop(0)
            if s < o:
                return True
            elif s > o:
                return False

        if not sv:
            return True

        return False


    def __str__(self):
        return ".".join(str(v) for v in self.version)


def solution(l):
    l = list(map(Version, l))
    l.sort()
    return list(map(str, l))




        

