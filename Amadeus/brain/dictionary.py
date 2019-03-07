# coding: utf-8
# Define a dict that Amadeus uses.

import pickle
import sys
import os
from Amadeus.brain.wordseg.base_wordseg import JiebaSeg


class Dictionary(dict):
    def write_to_file(self, path):
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            raise IOError

    def read_from_file(self, path):
        try:
            with open(path, "rb") as f:
                 self.update(pickle.load(f))
        except Exception as e:
            raise IOError


if __name__ == "__main__": 
    # argv: 1.a dir, all txt in it; 2.a file, to save the dict; 3. an int, top x words to keep
    inputs = sys.argv[1]
    output = sys.argv[2]
    topx = int(sys.argv[3])
    d = Dictionary()
    ws = JiebaSeg()
    for filename in os.listdir(inputs):
        try:
            with open(inputs + os.sep + filename) as f:
                for line in f:
                    for word in ws.cut(line):
                        d[word] = d.get(word, 0) + 1
        except Exception as e:
            print(e)
            continue
        print(filename + " done.")
    print("Wordseg has done.Now sort top " + str(topx))
    topkeys = sorted(d.keys(), key=lambda x: d[x], reverse=True)[:topx]
    # del more than new
    newd = Dictionary()
    for i, k in enumerate(topkeys):
        newd[k] = [i, d[k]]
    newd.write_to_file(output)
