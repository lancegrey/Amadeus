# coding: utf-8
# prepreocess conv


def preprocess_conv(f):
    ret = []
    for line in f.readlines():
        if line.strip() != "E":
            ret.append(line[1:].strip())
        if line.strip() == "E" and len(ret) > 0:
            yield [[x for i, x in enumerate(ret) if i%2 == 0], [x for i, x in enumerate(ret) if i%2 == 1]]
            ret = []

    # E as an ending
    # pass
