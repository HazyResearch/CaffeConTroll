import sys
import random

if len(sys.argv) != 2:
    print 'Usage: >>> python generate_conv_test.py test_name'
    sys.exit(0)

mB = 4
iD = 3
oD = 8
iR = 127
iC = 127
k = 11
s = 4
p = 2

test_name = sys.argv[1]

fname = 'input/conv_forward_in_' + test_name + '.txt'
f = open(fname, 'w')
print 'Creating ' + fname + '...'
for i in range(iR*iC*iD*mB):
    r = (1-2*random.random())/10
    f.write(str(r) + ' ')
f.close()

fname = 'input/conv_model_' + test_name + '.txt'
f = open(fname, 'w')
print 'Creating ' + fname + '...'
for i in range(k*k*iD*oD):
    r = (1-2*random.random())/10
    f.write(str(r) + ' ')
f.close()

fname = 'input/conv_bias_in_' + test_name + '.txt'
f = open(fname, 'w')
print 'Creating ' + fname + '...'
for i in range(oD):
    r = (1-2*random.random())/10
    f.write(str(r) + ' ')
f.close()

fname = 'input/conv_backward_model_' + test_name + '.txt'
f = open(fname, 'w')
print 'Creating ' + fname + '...'
for i in range(k*k*iD*oD):
    r = (1-2*random.random())/10
    f.write(str(r) + ' ')
f.close()

#fname = 'output/conv_forward_' + test_name + '.txt'
#f = open(fname, 'w')
#f.close()
#
#fname = 'output/conv_backward_' + test_name + '.txt'
#f = open(fname, 'w')
#f.close()
#
#fname = 'output/conv_bias_' + test_name + '.txt'
#f = open(fname, 'w')
#f.close()
#
#fname = 'output/conv_weights_' + test_name + '.txt'
#f = open(fname, 'w')
#f.close()
#

