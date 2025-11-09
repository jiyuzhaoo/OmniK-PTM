
import numpy as np
import scipy.io
import os


def PSTAAP_feature(protein_sequences, test_PSTAAP=False):
    for i in range(len(protein_sequences)):
        protein_sequences[i] = protein_sequences[i][:24] + protein_sequences[i][25:]

    if test_PSTAAP:
        mat_contents = scipy.io.loadmat("Feature_extraction_algorithms/Fr_test.mat")
    else:
        mat_contents = scipy.io.loadmat("Feature_extraction_algorithms/Fr_train.mat")

    Fr = mat_contents['Fr']
    """
    print(Fr[0*400+5*20+0,0])
    print(Fr[5 * 400 + 0 * 20 + 16, 1])
    print(Fr[0 * 400 + 16 * 20 + 14, 2])
    """
    AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    PSTAAP = np.zeros((len(protein_sequences), 46))
    for i in range(len(protein_sequences)):
        for j in range(len(protein_sequences[0])-2):
            t1 = protein_sequences[i][j]
            position1 = AA.index(t1)
            t2 = protein_sequences[i][j+1]
            position2 = AA.index(t2)
            t3 = protein_sequences[i][j+2]
            position3 = AA.index(t3)

            PSTAAP[i][j] = Fr[400 * position1 + 20 * position2 + position3][j]

    return PSTAAP
