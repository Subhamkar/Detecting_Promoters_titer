import numpy as np

def seq_length(seq_list):
    for i in range(len(seq_list)):
        seq = seq_list[i]
        length = len(seq)
    return length


def seq_matrix(seq_list, label):
    seq_len =seq_length(seq_list)
    tensor = np.zeros((len(seq_list), seq_len, 4))
    for i in range(len(seq_list)):
        seq = seq_list[i]


        j = 0
        for s in seq:
            if s == 'A':
                tensor[i][j] = [1, 0, 0, 0]
            if s == 'C':
                tensor[i][j] = [0, 1, 0, 0]
            if s == 'G':
                tensor[i][j] = [0, 0, 1, 0]
            if s == 'T':
                tensor[i][j] = [0, 0, 0, 1]

            j += 1
    if label == 1:
        y = np.ones((len(seq_list), 1))
    else:
        y = np.zeros((len(seq_list), 1))

    return tensor, y


def read_fasta(fp):
    name, seq = None, []
    for line in fp:
        line = line.rstrip()
        if line.startswith(">"):
            if name: yield (name, ''.join(seq))
            name, seq = line, []
        else:
            seq.append(line)
    if name: yield (name, ''.join(seq))
