# MT19937.py contains class definitions for MT19937 and MT19937_symbolic
#   Class `MT19937` is an implementation of MT19937. 
#     It does not have seed -> state bcuz different MT19937 implementations have different ways to generate state from seed
#     In addition, it also contains `MT19937.untwist` which untwists the state and `MT19937.reverse_states` which reverts state
#     It also implements basic MT19937 cloning and integrates well with `XorSolver`. Details in `MT19937.__init__`
#
#   Class `MT19937_symbolic` executes the prng algorithm MT19937 symbolically
#     This can be used for more general MT19937 cloning problems which class `MT19937` does not support

import random
import copy
import numpy as np
import gzip
import os

class MT19937():

    """MT19937 iterator. Inputs are internal state"""
    
    def __init__(self, state = None, ind = 0, state_from_solved_eqns = None, state_from_data = None):
        
        '''
        Init MT19937 constants
        
        If state is given, MT19937.state will be assigned said state
            state is a list of 624 32-bit numbers
        
        If state_from_solved_eqns is given, MT19937 will compute the state from given equations
            state_from_solved_eqns has to be fully solved
            state_from_solved_eqns is XorSolver.eqns upon calling XorSolver.solve or XorSolver.cryptominisat_solve
            
        If state_from_data is given, MT19937 will clone state from given data
            state_from_data has to be a tuple (data, nbits)
                data:
                    A list of numbers, each has nbits bits
                nbits:
                    A number (4, 8, 12, 16, 20, 24, 28, 32)
                    Represents number of bits each number has in data.
                    It is possible to clone for any number of bits given in range 1 to 32, but only the above numbers are supported
        '''
        
        self.NBIT, self.N, self.M, self.R = (32, 624, 397, 31)
        self.A = 0x9908B0DF
        self.U, self.D = (11, 0xFFFFFFFF)
        self.S, self.B = (7, 0x9D2C5680)
        self.T, self.C = (15, 0xEFC60000)
        self.L = 18
        
        self.lower_mask = (1 << self.R) - 1
        self.upper_mask = (2**self.NBIT - self.lower_mask - 1) % 2**self.NBIT
        
        self.untwist_mat = None
        
        self.ind = ind
        
        if (not state) and (not state_from_data) and (not state_from_solved_eqns):
            raise Exception("State of MT19937 not given. Provide either state or state_from_solved_eqns or state_from_data")
        if state_from_solved_eqns:
            self._init_state_from_eqns(state_from_solved_eqns)
        elif state_from_data:
            data, nbits = state_from_data
            self.clone_state_from_data(data, nbits)
        else:
            self.state = state
        
    @staticmethod
    def _apply_mat(A, mat, rhs = None):
        
        '''
        Multiplies A with mat (returns A*mat)
        Input matrix follows the matrix representation described in this class's docstring
        Multiplication is done mod 2 (i.e. addition can be seen as a xor operation)
        '''
        isset = False
        if type(A[0]) == set: isset = True
        
        def job(row):
            ele = False
            if isset: ele = set()
            for element in row:
                ele ^= A[element]
            return ele
            
        return [job(row) for row in mat]
        
    def _init_state_from_eqns(self, solved_eqns):
        
        '''State is initialised from solved equations'''
        
        solution = [None]*(self.N*self.NBIT)
        for idx in range(len(solved_eqns)):
            eqn_lhs = solved_eqns[idx][0]
            assert len(eqn_lhs) == 1, "solved_eqns is not fully solved."
            ind = list(eqn_lhs)[0]
            val = solved_eqns[idx][1]
            solution[ind] = val

        self.state = []
        for idx in range(self.N):
            x = 0
            for j in range(self.NBIT):
                x *= 2
                if solution[idx*self.NBIT + j]:
                    x += 1
            self.state.append(x)
            
    def _int2bits(self, y, bits = None):
        
        '''
        Converts a number y into an array of bools representing y's binary representation
        '''
        
        if not bits: bits = self.NBIT
        
        y_eq = bin(y)[2:]
        y_eq = '0'*(bits-len(y_eq)) + y_eq
        y_eq = [True if i=='1' else False for i in y_eq]
        
        return y_eq[:bits]
            
    def clone_state_from_data(self, data, nbits):
        
        '''
        Given data and nbits, computes state of cloned MT19937
        data:
            A list of numbers, each has nbits bits
            These numbers must be the first nbits of consecutive output from an MT19937
            The number of numbers from this list should be at least 624 * 32//nbits to have enough info to clone state.
        nbits:
            A number (4, 8, 12, 16, 20, 24, 28, 32)
            Represents number of bits each number has in data.
            It is possible to clone for any number of bits given in range 1 to 32, but only the above numbers are supported
            
        This works via loading a precomputed inverse matrix and applying it to data
        '''
        
        # Check whether value of nbits is supported
        supported_nbits = [4, 8, 12, 16, 20, 24, 28, 32]
        assert nbits in supported_nbits, "{} nbits is not supported. Only {} are supported".format(nbits, supported_nbits)
        
        nb = nbits
        iterr = int(31.9//nb + 1)
        ndata = self.N*iterr
        
        # Check if sufficient data is given
        assert len(data) >= ndata, "Data given is not enough to clone state. You gave {} numbers, required {}.".format(len(data), ndata)
        data = data[:ndata]
        
        # Load matrices
        # solve_mat is the matrix used to solve the equations
        # verify_mat is to check if data is indeed the first nbits bits of consecutive output of an MT19937
        inv_filename = os.path.dirname(__file__) + "/../matrices/inverse_nb-{}_iterr-{}_ndata-{}.npy.gz".format(nb, iterr, ndata)
        ver_filename = os.path.dirname(__file__) + "/../matrices/verify_nb-{}_iterr-{}_ndata-{}.npy.gz".format(nb, iterr, ndata)
        f = gzip.GzipFile(inv_filename, "r")
        solve_mat = np.load(f, allow_pickle=True)
        f = gzip.GzipFile(ver_filename, "r")
        verify_mat = np.load(f, allow_pickle=True)
        f.close()
        
        # Form rhs of equations
        eqns_rhs = []
        for n in data:
            eqns_rhs.extend(self._int2bits(n, bits=nbits))
        
        # Verify data
        for row in verify_mat:
            b = False
            for idx in row:
                b ^= eqns_rhs[idx]
            assert b == False, "data is not consecutively generated from MT19937 or you're using this function wrongly"
        
        # Solve for state (cloning state)
        state_bool = self._apply_mat(eqns_rhs, solve_mat)

        # Reconstruct state from state_bool
        self.state = []
        for idx in range(self.N):
            x = 0
            for j in range(self.NBIT):
                x *= 2
                if state_bool[idx*self.NBIT + j]:
                    x += 1
            self.state.append(x)
    
    def twist(self, state):
        
        '''twist the states'''
        
        for i in range(self.N):
            # x = states[(i+1) % n] (except highest bit) + states[i] (highest bit)
            x = (state[i] & self.upper_mask) + (state[(i+1) % self.N] & self.lower_mask)
            
            xA = x >> 1
            if (x % 2) != 0: # invokes if states[(i+1) % n] has 0-th bit set
                xA = xA ^ self.A
                
            state[i] = state[(i + self.M) % self.N] ^ xA
            
        return state
    
    def untwist(self, state):
        
        '''
        untwist the states, but since it can't recover state[0]
            state is a list of 624 32-bit numbers
        '''
        
        #original_state = state.copy()
        
        if type(self.untwist_mat) == type(None):
            filename = os.path.dirname(__file__) + "/../matrices/untwist.npy.gz"
            f = gzip.GzipFile(filename, "r")
            self.untwist_mat = np.load(f, allow_pickle=True)
            
        # Form rhs of equations
        eqns_rhs = []
        for n in state:
            eqns_rhs.extend(self._int2bits(n, bits=self.NBIT))
            
        # Untwist
        state_bool = self._apply_mat(eqns_rhs.copy(), self.untwist_mat)
        
        # Reconstruct state from state_bool except state[0]
        state = []
        for idx in range(self.N):
            x = 0
            for j in range(self.NBIT):
                x *= 2
                if state_bool[idx*self.NBIT + j]:
                    x += 1
            state.append(x)
            
        return state
    
    def _shift_states_backwards_623(self, states):
        
        '''
        states must contain 625 entries (n+1)
        If given states [0,1,2...,624]
        State returned is 
        [0-623, 1-623..., 624-623] = [-623, -622..., 1]
        '''
        
        prev_state = self.untwist(states[1:])
        prev_state[0] = self.untwist(states[:-1])[1]
        prev_state = prev_state + [states[1]]
        return prev_state
        
    def reverse_states(self, n_steps):
        
        '''
        [WIP] Reverse MT19937 states by n_steps
        
        Only works sometimes, idk why it doesnt work on certain states.
        This can also be about 2x faster with more code but since it still
        doesn't work all the time I didn't write it.
        '''
        
        assert n_steps >= 0, "Can only reverse a positive number of states"
        
        states_625 = self.state + [self.twist(self.state)[0]]
        
        while True:
            n_steps -= self.N-1
            states_625 = self._shift_states_backwards_623(states_625.copy())
            if self.ind > n_steps:
                break
                
        if self.ind != 0:
            states_625 = self._shift_states_backwards_623(states_625.copy())
            self.ind -= 1
            
        self.state = states_625[:-1]
        self.ind -= n_steps
    
    def generate(self, ind, states):
        
        '''generate random number from states'''
        
        if ind == 0:
            states = self.twist(states)

        y = states[ind]
        y = y ^ ((y >> self.U) & self.D)
        y = y ^ ((y << self.S) & self.B)
        y = y ^ ((y << self.T) & self.C)
        y = y ^  (y >> self.L)
        
        return y % 2**self.NBIT, states
    
    def __call__(self):
        
        '''Returns random number and updates index'''
        
        y, self.state = self.generate(self.ind, self.state)
        self.ind = self.ind + 1
        self.ind %= self.N
        return y
    
class MT19937_symbolic():
    
    '''
    Executes MT19937 symbolically
    
        States is represented as an array of sets of size (N * NBIT), each set represents a bit
            Each block of size NBIT represents a NBIT-bit number of the state
            Starting from MSB to LSB
            Original state is represented as [{0}, {1}, {2}... {N*NBIT-1}],
            where {i} represents the i-th bit of the state (a_i)
            
        Output is an array of sets of size NBIT, each set represents a bit
            Starting from MSB to LSB
            E.g. 
                if out = [{1, 3}, {4, 1} ...],
                MSB of output is a_1 ^ a_3 and 2nd MSB is a_4 ^ a_1  
            
        A matrix is represented as an array of sets. 
        States and output described above can hence be seen as matrices
            E.g.
                if mat = [{1, 3}, {4, 1}]
                It can be seen as a matrix where the 1st row contain 1s in the 1st and 3rd position only.
                And the 2nd row contain 1s in the 4th and 1st position only
    '''
    
    def __init__(self, ind = 0):
        
        '''
        Init MT19937 constants,
        Init variables used for symbolic computation
        Init matrix for twist and generate operation for faster symbolic computation
        '''
        
        self.NBIT, self.N, self.M, self.R = (32, 624, 397, 31)
        self.A = 0x9908B0DF
        self.U, self.D = (11, 0xFFFFFFFF)
        self.S, self.B = (7, 0x9D2C5680)
        self.T, self.C = (15, 0xEFC60000)
        self.L = 18
        
        self.lower_mask = (1 << self.R) - 1
        self.upper_mask = (2**self.NBIT - self.lower_mask - 1) % 2**self.NBIT
        
        self.state_sym = [{i} for i in range(self.NBIT * self.N)]
        self.IDX = ind
        self.bin_a = self._int2bits(self.A)
        self.bin_b = self._int2bits(self.B)
        self.bin_c = self._int2bits(self.C)
        
        # For speedup, make these repeated computation is a better form
        self.twist_mat = copy.deepcopy(self._twist_sym_init(copy.deepcopy(self.state_sym)))
        self.generate_mat = copy.deepcopy(self._generate_sym_init([{i} for i in range(self.NBIT)]))
        
    @staticmethod
    def _apply_mat(A, mat):
        
        '''
        Multiplies A with mat (returns A*mat)
        Input matrix follows the matrix representation described in this class's docstring
        Multiplication is done mod 2 (i.e. addition can be seen as a xor operation)
        '''
        
        def job(row):
            ele = set()
            for element in row:
                ele ^= A[element]
            return ele
            
        return [job(row) for row in mat]
    
    def _int2bits(self, y, bits = None):
        
        '''
        Converts a number y into an array of bools representing y's binary representation
        '''
        
        if not bits: bits = self.NBIT
        
        y_eq = bin(y)[2:]
        y_eq = '0'*(bits-len(y_eq)) + y_eq
        y_eq = [True if i=='1' else False for i in y_eq]
        
        return y_eq[:bits]
    
    def _twist_sym_init(self, state_sym):
        
        '''Generates the twist matrix'''
        
        for i in range(self.N):

            # x = (states[i] & upper_mask) + (states[(i+1) % N] & lower_mask)
            x_sym = [state_sym[i*self.NBIT]] + state_sym[((i+1) % self.N) * self.NBIT + 1 : ((i+1) % self.N + 1) * self.NBIT]
            x_sym = copy.deepcopy(x_sym)

            # xA = x >> 1
            # if (x % 2) != 0:
            #     xA = xA ^ A
            xA_sym = [set()] + x_sym[:-1]
            xLSB_sym = x_sym[-1]

            for j in range(self.NBIT):
                if self.bin_a[j]:
                    xA_sym[j] ^= xLSB_sym

            # states[i] = states[(i + M) % N] ^ xA
            for j in range(self.NBIT):
                state_sym[i*self.NBIT + j] = state_sym[((i+self.M)%self.N)*self.NBIT + j] ^ xA_sym[j]
                
        return state_sym
    
    def _twist_sym(self):
        
        '''Twist states symbolically'''
        
        self.state_sym = self._apply_mat(self.state_sym, self.twist_mat)
        
    def _generate_sym_init(self, y_sym):
        
        '''Generates the generate matrix'''
        
        # y = y ^ ((y >> self.U) & self.D)
        for i in range(self.NBIT-1, self.U-1, -1):
            y_sym[i] ^= y_sym[i-self.U]

        # y = y ^ ((y << self.S) & self.B)
        for i in range(self.NBIT-self.S):
            if self.bin_b[i]:
                y_sym[i] ^= y_sym[i+self.S]

        # y = y ^ ((y << self.T) & self.C)
        for i in range(self.NBIT-self.T):
            if self.bin_c[i]:
                y_sym[i] ^= y_sym[i+self.T]

        # y = y ^  (y >> self.L)
        for i in range(self.NBIT-1, self.L-1, -1):
            y_sym[i] ^= y_sym[i-self.L]
            
        return y_sym
    
    def _generate_sym(self):
        
        '''Generates random number based on states symbolically'''
                
        if self.IDX == 0:
            self._twist_sym()

        y_sym = self.state_sym[self.IDX*self.NBIT: (self.IDX+1)*self.NBIT]
            
        return self._apply_mat(y_sym, self.generate_mat)
    
    def cast_num(self, original_state, num_sym):
        
        '''
        Cast number with a symbolic representation:
        
            Input:
                original_state: 
                    original state of MT19937
                    An array of NBIT-bit numbers of size N
                num_sym:
                    Symbolic representation of the number
                    An array of sets, each set represents a bit of the number
             
             Output:
                 32-bit number
        '''
        
        states_eq = []
        for num in original_state:
            states_eq.extend(self._int2bits(num))
            
        x_out = 0
        for b in num_sym:
            bit = False
            for e in b:
                bit ^= states_eq[e]
            x_out = 2*x_out + int(bit)
            
        return x_out
    
    def cast_state(self, original_state):
        
        '''
        Resolves the original_state with the symbolic current state (self.state_sym):
        
            Input:
                original_state: 
                    original state of MT19937
                    An array of NBIT-bit numbers of size N
             
            Output:
                An array of 32-bit numbers which is the current state
        '''
        
        states_eq = []
        for num in original_state:
            states_eq.extend(self._int2bits(num))

        states_out = []
        for idx in range(self.N):
            x_out = 0
            for b in self.state_sym[idx*self.NBIT : (idx+1)*self.NBIT]:
                bit = False
                for e in b:
                    bit ^= states_eq[e]
                x_out = 2*x_out + int(bit)
            states_out.append(x_out)
        
        return states_out
    
    def __call__(self):
        
        '''Returns symbolic representation of random number and increments index'''
        
        y_sym = self._generate_sym()
        self.IDX = self.IDX + 1
        self.IDX %= self.N
        return y_sym
    
if __name__ == "__main__":
    
    """
    Verifies the symbolic execution of MT19937
    """
    
    print("Testing the validity of MT19937_symbolic")
    
    original_state = [random.randint(0, 2**32) for _ in range(624)]

    state = original_state.copy()
    rng = MT19937(state, ind = 0)
    rng_sym = MT19937_symbolic(ind = 0)

    for _ in range(10000):
        
        y_eq = rng()
        y_sym = rng_sym()
        y_out = rng_sym.cast_num(original_state, y_sym)
        
        assert(y_eq == y_out)