
print """
#######SPEED UP#######
Single NUMA Node
Increase # Partition
Each Parititon uses 1 Core
Input = 64 x 64  Kernel = 5 x 5  Channels: Input: 96 Output: 256  BatchSize = 256
######################
"""

import os

os.system('mkdir -p bin')
for p in [1,2,4,8,16,32]:
    for t in [1,]:
        os.system('NPARTITION=%d NTHREADPERPARTITION=%d make speedup -j8' % (p, t))

os.system('mkdir -p rs')
for p in [1,2,4,8,16,32]:
    for t in [1,]:
        os.system('numactl --cpubind=0 --membind=0 bin/speedup.%d.%d | tee rs/speedup.%d.%d' % (p,t,p,t))
