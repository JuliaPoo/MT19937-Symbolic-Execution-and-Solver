# XorSolver.py contains class definition for XorSolver, a general purpose solver for matrices over GF(2)
# Includes a python wrapper for Cryptominisat and a python-only solver

import subprocess
import os

def execute(cmd):
    
    '''Executes cmd and is a generator for cmd output'''
    
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        return return_code

RESERVED = {-1}

class XorSolver():
    
    def __init__(self, eqns, nvars):
        
        '''
        Initialise Xor_solver
        eqns:
            System of equations to solve.
            In the format of a list of eqn
                Each eqn is a list containing a set and a number, denoting LHS and RHS.
                E.g.
                [{0,2,3}, 1] represents the equation a0 ^ a2 ^ a3 = 1
                Variables uses index from 0 to nvars-1
        nvars:
            Number of variables in the system of equations
        '''
        
        self.eqns = eqns
        self.nvars = nvars
        
    
    @staticmethod
    def _remove_reserved_bit_eqn(eqn):
        
        '''
        Removes 1 from the LHS of eqn by xor both sides with 1
        1 is represented as index -1, and this is defined in the global RESERVED
        '''
        
        res = list(RESERVED)[0]
        if res in eqn[0]: 
            eqn[0] = eqn[0] ^ RESERVED
            eqn[1] = eqn[1] ^ True
        return eqn
    
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
    
    def _check_consistency_eqn(self, eqn):
        '''
        Check if an empty equation equates to 1
        Used only upon achieving RREF of matrix
        '''
        assert not (len(eqn[0]) == 0 and eqn[1] == 1), "solve_xor: [ERROR] System of equations is inconsistent and unsolvable"
        
    def _remove_reserved_bit(self):
        '''
        Removes 1 from all equations' LHS
        '''
        res = list(RESERVED)[0]
        for idx, eqn in enumerate(self.eqns):
            self.eqns[idx] = self._remove_reserved_bit_eqn(eqn)
        
    def _check_consistency(self):
        '''
        Checks consistency of whole system of eqns
        Assumes self.eqns is in RREF form
        '''
        for eqn in self.eqns:
            self._check_consistency_eqn(eqn)
            
    def _remove_empty_eqns(self):
        '''
        Removes empty equations from self.eqns
        '''
        self.eqns = [i for i in self.eqns if len(i[0]) != 0]
    
    def solve(self, cleanup = True, verbose = True):
        
        '''
        Solves self.eqns
        If self.eqns not satisfiable, self.eqns is the RREF
        
        cleanup:
            If True, will remove empty equations from RREF
            
        verbose:
            Prints progress
        '''
        
        # Remove RESERVED
        self._remove_reserved_bit()
        
        # Solve
        var = 0
        used = []
        for var in range(self.nvars):
            
            # Index of all eqns containing vars
            column = [i for i,row in enumerate(self.eqns) if var in row[0]]
            
            # Get element that isnt in used and has minimum length and select that. Update used.
            l_list = [(len(self.eqns[ieqn][0]), ieqn) for ieqn in column if ieqn not in used]
            if len(l_list) == 0: continue
            selected = min(l_list, key=lambda x: x[0])[1]
                    
            used.append(selected)        
            
            # Do row operations
            for i in column.copy():
                if i == selected:
                    continue
                
                w_eqn = self.eqns[i]
                r_eqn = self.eqns[selected]
                
                w_eqn[0] ^= r_eqn[0]
                w_eqn[1] ^= r_eqn[1]
                
            #print("{}, {}, {}         \r".format(var, len(column), len(w_eqn[0])), end="")
            if verbose:
                print("{}/{}   \r".format(var+1, self.nvars), end="")
            
        self._check_consistency()
        if cleanup: self._remove_empty_eqns()
            
    def cryptominisat_solve(self, cryptominisat_path, cnf_file, n_threads = 8, verbose = 1, maxmatrixrows = 5000):
        
        '''
        Python wrapper for cryptominisat
            cryptominisat_path: Path to cryptominisat executable
            
            cnf_file: 
                filename that will be used to store .cnf file which is input to cryptominisat
                
            n_threads: 
                Number of threads used
                
            verbose: 
                Integer from 0 to 4. Determine cryptominisat output's verbosity
                
            maxmatrixrows: 
                Maximum number of rows cryptominisat will consider in a xor matrix
            
        To use this method, you must have built cryptominisat with Gaussian Elimination, ideally with m4ri for speed.
        https://github.com/msoos/cryptominisat
        
        I did not speed test this with m4ri but without m4ri solving a dense matrix with size 19968 takes about an hour
        '''
        
        nvars = self.nvars
        nclauses = len(self.eqns)

        # Checking and removing cnf_file if it already exists
        if os.path.isfile(cnf_file):
            os.remove(cnf_file)

        # Write eqns to .cnf file
        with open(cnf_file, 'a') as f:

            # Write header
            f.write("p cnf {} {}\n".format(nvars, nclauses))

            # Write eqns
            for eqn in self.eqns:
                buffer = " ".join([str(i+1) for i in eqn[0]]) + " 0"
                if eqn[1] == False:
                    buffer = "-" + buffer
                buffer = "x" + buffer
                f.write(buffer + "\n")
        
        # Running cryptominisat
        cryptominisat_cmd = "{} --verb={} --threads={} --maxmatrixrows={} {}"
        cryptominisat_cmd = cryptominisat_cmd.format(cryptominisat_path, verbose, n_threads, maxmatrixrows, cnf_file)

        if verbose: print("[*] Executing command ", cryptominisat_cmd)
        
        solution_str = ""
        for line in execute(cryptominisat_cmd):
            if line[0] == 'v':
                solution_str += line
            if verbose:
                print(line, end="")
        
        error_msg = "Something went wrong with executing {}. Ensure that you have built cryptominisat with gaussian elimination and the paths you provided are correct."
        assert len(solution_str) != 0, error_msg.format(cryptominisat_cmd)
        
        # Reconstruct solution and save it in self.eqns
        solution_list = solution_str.replace("\n", "").replace("v", "").replace("  ", " ").strip().split(" ")
        self.eqns = []
        for line in solution_list[:-1]:
            n = int(line)
            self.eqns.append([{abs(n)-1}, n > 0])
        