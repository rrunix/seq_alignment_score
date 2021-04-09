from cython.parallel import prange
from libc.stdlib cimport malloc, free
from libc.string cimport memset
cimport numpy as np
import numpy as np


cdef inline double fast_calc_affine_penalty(int length, double open, double extend, int penalize_extend_when_opening) nogil:
    """
    Calculate a penality score for the gap function. 
    Implementation from https://github.com/biopython/biopython/blob/master/Bio/pairwise2.py
    
    """
    if length <= 0:
        return 0.0
    cdef double penalty = open + extend * length
    if penalize_extend_when_opening == 0:
        penalty -= extend
    return penalty


cdef inline double getCell(double * array, int col_size, int row, int col) nogil:
    cdef int row_cycle = row % 2
    return array[row_cycle * col_size + col]


cdef inline void setCell(double * array, int col_size, int row, int col, double value) nogil:
    cdef int row_cycle = row % 2
    array[row_cycle * col_size + col] = value


cdef float fast_score(
    long[:] sequenceA,
    long[:] sequenceB,
    double match,
    double mismatch,
    double open,
    double extend,
    int penalize_extend_when_opening
    ) nogil:
    """
    Calculate the score of the Needleman-Wunsch aligment (based on Gotoh modification).
    Implementation based on https://github.com/biopython/biopython/blob/master/Bio/pairwise2.py

    Args:
        sequenceA (long[]): A sequence of numbers.
        sequenceB (long[]): Another sequence of numbers.
        match (double): score if two elements match (positive).
        mismatch (double): score if two elemnts are different (negative).
        open (double): open gap score (negative).
        extend (double): extending gap score (negative and less or equals than open).
        penalize_extend_when_opening (int): Whether to add the extend penalization when opening a gap (0 no penalization, 1 penalization).

    Returns:
        The Needleman-Wunsch aligment score.
    """
    
    cdef double first_A_gap = fast_calc_affine_penalty(1, open, extend, penalize_extend_when_opening)
    cdef double first_B_gap = fast_calc_affine_penalty(1, open, extend, penalize_extend_when_opening)
    # Create the score and traceback matrices. These should be in the
    # shape:
    # sequenceA (down) x sequenceB (across)
    cdef int lenA = len(sequenceA)
    cdef int lenB = len(sequenceB)
    cdef int col_size = lenB + 1

    cdef double* score_matrix = <double *> malloc(2 * (lenB + 1) * sizeof(double))
    memset(score_matrix, 0, 2 * (lenB + 1) * sizeof(double))
    
    # Now initialize the col 'matrix'. Actually this is only a one dimensional
    # list, since we only need the col scores from the last row.
    cdef double* col_score = <double *> malloc((lenB + 1) * sizeof(double))
    memset(col_score, 0, (lenB + 1) * sizeof(double))
    
    cdef int i
    for i in range(1, lenB + 1):
        col_score[i] = fast_calc_affine_penalty(i, 2 * open, extend, penalize_extend_when_opening)

    # The row 'matrix' is calculated on the fly. Here we only need the actual
    # score.
    # Now, filling up the score and traceback matrices:
    
    cdef int row, col
    cdef double match_score, row_score, nogap_score, row_open, row_extend, col_open, col_extend, best_score
    
    for row in range(1, lenA + 1):
        row_score = fast_calc_affine_penalty(
            row, 2 * open, extend, penalize_extend_when_opening
        )
        for col in range(1, lenB + 1):
            # Calculate the score that would occur by extending the
            # alignment without gaps.

            if sequenceA[row - 1] == sequenceB[col - 1]:
                match_score = match
            else:
                match_score = mismatch

            nogap_score =  getCell(score_matrix, col_size, row - 1, col - 1) + match_score

            # Check the score that would occur if there were a gap in
            # sequence A. This could come from opening a new gap or
            # extending an existing one.
            # A gap in sequence A can also be opened if it follows a gap in
            # sequence B:  A-
            #              -B
            
            # score_matrix[row][col - 1]

            if row == lenA:
                row_open = getCell(score_matrix, col_size, row, col - 1)
                row_extend  = row_score
            else:
                row_open = getCell(score_matrix, col_size, row, col - 1) + first_A_gap
                row_extend = row_score + extend

            row_score = max(row_open, row_extend)

            # The same for sequence B:
            # score_matrix[row - 1][ col]
            if col == lenB:
                col_open = getCell(score_matrix, col_size, row - 1, col)
                col_extend = col_score[col]
            else:
                col_open = getCell(score_matrix, col_size, row - 1, col) + first_B_gap
                col_extend = col_score[col] + extend

            col_score[col] = max(col_open, col_extend)

            best_score = max(nogap_score, col_score[col], row_score)
            # score_matrix[row][ col]
            setCell(score_matrix, col_size, row, col, best_score)


    free(score_matrix)
    free(col_score)
    return best_score


cdef _matrix_scores_impl(seqs_a, seqs_b, match, mismatch, open, extend, penalize_extend_when_opening):

    cdef long[:, :] _seqs_a = np.array(seqs_a, dtype=int)
    cdef long[:, :] _seqs_b = np.array(seqs_b, dtype=int)
    cdef double _match = match
    cdef double _mismatch = mismatch
    cdef double _open = open
    cdef double _extend = extend
    cdef int _penalize_extend_when_opening = int(penalize_extend_when_opening)
    
    cdef int lenA = len(seqs_a)
    cdef int lenB = len(seqs_b)
    cdef double[:, :] scores = np.zeros((len(seqs_a), len(seqs_b)))
    
    cdef int i, j
    cdef int symmetric = 0

    if lenA == lenB:
        symmetric = int(np.all(seqs_a == seqs_b))

    if symmetric == 1:
        for i in prange(lenA, nogil=True):
            for j in range(i, lenB):
                scores[i, j] = scores[j, i] = fast_score(_seqs_a[i], _seqs_b[j], _match, _mismatch, 
                    _open, _extend, _penalize_extend_when_opening)
    else:
        for i in prange(lenA, nogil=True):
            for j in range(lenB):
                scores[i, j] = fast_score(_seqs_a[i], _seqs_b[j], _match, _mismatch, 
                    _open, _extend, _penalize_extend_when_opening)
            
    return np.asarray(scores).reshape((len(seqs_a), len(seqs_b)))


def nw_score(seq_a, seq_b, match, mismatch, open, extend, penalize_extend_when_opening):
    """
    Calculate the score of the Needleman-Wunsch aligment (based on Gotoh modification).

    Args:
        seq_a (int[]): A sequence of numbers.
        seq_b (int[]): Another sequence of numbers.
        match (double): score if two elements match (positive).
        mismatch (double): score if two elemnts are different (negative).
        open (double): open gap score (negative).
        extend (double): extending gap score (negative and less or equals than open).
        penalize_extend_when_opening (bool): Whether to add the extend penalization when opening a gap.

    Returns:
        The Needleman-Wunsch aligment score.
    """
    _seq_a = np.array(seq_a, dtype=int)
    _seq_b = np.array(seq_b, dtype=int)
    _match = float(match)
    _mismatch = float(mismatch)
    _open = float(open)
    _extend = float(extend)
    _penalize_extend_when_opening = int(penalize_extend_when_opening)
    scores = fast_score(_seq_a, _seq_b, _match, _mismatch, _open, _extend, _penalize_extend_when_opening)
    return scores


def nw_score_matrix(seqs_a, seqs_b, match, mismatch, open, extend, penalize_extend_when_opening):
    """
    Calculate the score of the Needleman-Wunsch aligment (based on Gotoh modification). Calculates the pairwise
    scores of the sequences in seqs_a with the sequences in seqs_b.

    Args:
        seqs_a (int[][]): A list of sequences of numbers.
        seqs_b (int[][]): Another list of sequences of numbers.
        match (double): score if two elements match (positive).
        mismatch (double): score if two elemnts are different (negative).
        open (double): open gap score (negative).
        extend (double): extending gap score (negative and less or equals than open).
        penalize_extend_when_opening (int): Whether to add the extend penalization when opening a gap (0 no penalization, 1 penalization).

    Returns:
        A len(seqs_a) by len(seqs_b) matrix where the position i,j is the Needleman-Wunsch aligment score of seqs_a[i] with seqs_b[j].
    """
    return _matrix_scores_impl(seqs_a, seqs_b, match, mismatch, open, extend, penalize_extend_when_opening)
