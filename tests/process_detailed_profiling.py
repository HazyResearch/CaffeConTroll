# First, run:
# >>> grep -B 1 "Time Elapsed" profile.out > input_to_this_script.txt


import sys
if len(sys.argv) != 2:
    print 'Usage: >>> python process.py filename'
    sys.exit(0)

f = open(sys.argv[1])
current_layer = ''
layer_to_time = {}
for line in f:
    line = line.strip()
    if '--' in line:
        pass
    elif 'Data Throughput' in line:
        pass
    elif 'Time Elapsed' in line:
        layer_to_time[current_layer].append(float((line.split())[3]))
    elif 'REPORT' in line:
        if line not in layer_to_time.keys():
            layer_to_time[line] = []
        current_layer = line

for k in layer_to_time.keys():
    time_list = layer_to_time[k]
    total_sum = sum(time_list)
    total_num = len(time_list)
    mean = total_sum / float(total_num)
    print k + "\t" + str(total_num) + "\t" + str(mean)

f.close()
