import csaps
import numpy as np
import ruptures as rpt
from ruptures.base import BaseCost
import time
import cProfile
import pstats
import math


class CSSD_baseline(BaseCost):

    min_size = 1
    jump = 1

    def fit(self, signal):
        self.signal = signal
        return self

    def error(self, start, end):
        x = self.x[start: end]
        if len(x) <= 2:
            return 0
        
        signal = self.signal[start:end]
        weights = self.weights[start:end]
        
        # fit a spline
        spline = csaps.csaps(x, signal, smooth=self.p, weights=weights)
        
        # computing the data error
        data_err = np.sum(weights * (signal - spline(x))**2)
        
        # computing the inner energy
        pp = spline.spline
        ddpp = pp.derivative(nu=2)
        h = np.diff(ddpp.x)
        # Evaluate the piecewise polynomial at the breakpoints (excluding the last breakpoint)
        l0 = ddpp(ddpp.x[:-1])
        # Evaluate the piecewise polynomial at the breakpoints (excluding the first breakpoint)
        lh = ddpp(ddpp.x[1:])
        # Calculate the energy
        enSmooth = np.sum(h * (l0**2 + l0 * lh + lh**2)) / 3
        
        # sum both energiess
        energy = self.p * data_err +  (1- self.p) * enSmooth

        self.num_calls += 1
        self.num_elems += len(x)
        return energy
    
    def model(self):
        return "CSSD_baseline"

def detect_changepoints(x,y, p, gamma, delta):

    # start and time main algorithm
    start_time = time.time()
    
    # set up the custom costs
    cost = CSSD_baseline()
    cost.p = p
    cost.weights = np.power(delta, (-2))
    cost.x = x
    cost.num_calls = 0
    cost.num_elems = 0
    
    # invoke PELT method
    if math.isinf(gamma):
        result = []
    else:
        algo = rpt.Pelt(custom_cost=cost, min_size=1, jump=1).fit(y)
        result = algo.predict(pen=gamma)
    elapsed_time = time.time() - start_time
    
    return result, elapsed_time, cost.num_calls, cost.num_elems

# a simple example function call
def sample_call():
    n = 200
    x = np.linspace(0,1,n)
    y = np.array(x > 0.5, dtype=float) + 0.01*np.random.randn(n)
    result, time = detect_changepoints(x, y, 0.9999, 0.1, 0.01*np.ones(x.shape))
    print(f'{result=}, {time=}')

if __name__ == '__main__':
    cProfile.run('sample_call()', 'profiling_results.out')
    
    # Create a pstats.Stats object from the profiling results file
    stats = pstats.Stats('profiling_results.out')

    # Sort the results by cumulative time and print the top 10 functions
    stats.sort_stats('cumtime').print_stats(20)


