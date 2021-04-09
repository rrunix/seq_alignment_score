# Sequence metrics: 
A bunch of pairwise sequence metrics. The algorithms are implemented in cython (fast).

Currently, the following algorithms are implemented:
* Needleman-Wunsch aligment score.

Installation:
 * Download the code: git clone https://github.com/rrunix/sequence_metrics.git
 * Install: python setup.py install (Currently, it requires Cython to be installed)

Usage:
```python

from sequence_metrics.alignment import nw_score, nw_matrix_score

# Two sequences
print(nw_score([0, 1, 1], [0, 1, 1, 2], 2, -1, -0.5, -0.1, True))
# Prints 6.0

# Two list of sequences
print(nw_matrix_score([[0, 1, 1], [3,4,5]], [[0, 1, 1, 2]], 2, -1, -0.5, -0.1, True))
# Prints array([[6.], [0.]])
```
