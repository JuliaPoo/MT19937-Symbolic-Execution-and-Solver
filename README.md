# MT19937 Symbolic Execution and Solver

This provides a class that performs symbolic execution of MT19937, as well as a solver for GF(2) matrices and precomputed solutions for certain instances of MT19937 cloning. The solver has a Python-Only solver with no other dependencies and a python wrapper for [Cryptominisat](https://github.com/msoos/cryptominisat) to solve GF(2) matrices. Note that Cryptominisat has to be built with GAUSS.

The Python-Only solver is faster than Cryptominisat built without [M4RI](https://github.com/malb/m4ri) but takes up a lot of RAM.

## What it is

A demo of all the main features are present in the python notebook [Demo of Features.ipynb](https://github.com/JuliaPoo/MT19937-Symbolic-Execution-and-Solver/blob/master/Demo%20of%20Features.ipynb)

There are three ways to clone MT19937 provided here. The first is using one of the precomputed solutions, which are used if the known numbers are the 4\*k most significant bits of consecutive outputs of an MT19937 generator, e.g python's `random.getrandbits(nbits)`. This usually solves in 1-2 seconds. The next two are with the Python-Only solver and Cryptominisat. The Python-Only solver takes around a minute for sparse matrices but up to two hours for dense matrices. I did not do tests with Cryptominisat as I wasn't able to build Cryptominisat with M4RI on Windows. I suggest building Cryptominisat with M4RI for the speed. Regardless, a Windows x64 built of Cryptominisat with GAUSS but no M4RI is included in the bin folder.

This also provides a way to reverse the state of an MT19937 generator, to predict numbers generated before the known numbers. However, as of now there is a bug somewhere that makes it work only sometimes. In the event that reversing does not work, simply forward the cloned MT19937 and reverse.

These are all demonstrated in [Demo of Features.ipynb](https://github.com/JuliaPoo/MT19937-Symbolic-Execution-and-Solver/blob/master/Demo%20of%20Features.ipynb)

## Dependencies

```
numpy
gzip
```
Do note to use the Cryptominisat Python wrapper you are required to first built Cryptominisat with GAUSS. For speed it should also be built with M4RI.
