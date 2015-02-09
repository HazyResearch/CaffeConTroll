
print """
#######SPEED UP#######
Single NUMA Node
Single Partition
Increase Batch Size
Increase # Thread Per Batch
Input = 64 x 64  Kernel = 5 x 5  Channels: Input: 96 Output: 256  BatchSize = 256
######################
"""

import os

os.system('mkdir -p bin')
for b in [1,8,64]:
    for t in [1,4,8]:
        os.system('NBATCH=%d NTHREAD=%d make batch -j8' % (b, t))

os.system('mkdir -p rs')
for b in [1,8,64]:
    for t in [1,4,8]:
        os.system('numactl --cpubind=0 --membind=0 bin/batch.%d.%d | tee rs/batch.%d.%d' % (b,t,b,t))
