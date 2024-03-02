import sys


class HypothesisBuffer:

    def __init__(self):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []

        self.last_commited_time = 0
        self.last_commited_word = None

    def insert(self, new, offset):
        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a,b,t) for a,b,t in new if a > self.last_commited_time-0.1]
        if self.new:
            a,b,t = self.new[0]
            if abs(a - self.last_commited_time) < 1 and self.commited_in_buffer:
                # it's going to search for 1, 2, ..., 5 consecutive words (n-grams) that are identical in commited and new. If they are, they're dropped.
                cn = len(self.commited_in_buffer)
                nn = len(self.new)
                for i in range(1,min(min(cn,nn),5)+1):  # 5 is the maximum 
                    c = " ".join([self.commited_in_buffer[-j][2] for j in range(1,i+1)][::-1])
                    tail = " ".join(self.new[j-1][2] for j in range(1,i+1))
                    if c == tail:
                        print("removing last",i,"words:",file=sys.stderr)
                        for _ in range(i):
                            print("\t",self.new.pop(0),file=sys.stderr)
                        break

    def flush(self):
        commit = []
        while self.new:
            na, nb, nt = self.new[0]
            if len(self.buffer) == 0:
                break
            elif nt == self.buffer[0][2]:
                commit.append((na,nb,nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
            
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer
