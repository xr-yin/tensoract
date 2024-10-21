"""
regarding device and data type choice

psi should be initialize on the target device and target dtype
e.g. cuda and complex64, cpu and complex128. Note, use complex type 
right from the start 

model should be initialize on the cpu with dtype=double, this is to ensure
accurate construction of the unitary MPO and Kraus operators during the 
simulation. Since the construction is only done once in a simulation, the 
computational cost is not a concern. A conversion can be done in the simulator 
(e.g. LindbladOneSite).

psi should be treated as a static variable 
"""

from .core import *
from .models import *
from .solvers import *