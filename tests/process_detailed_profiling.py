import operator
import sys
if len(sys.argv) != 2:
    print 'Usage: >>> python process.py filename'
    sys.exit(0)

total_num_iters = 0
f = open(sys.argv[1])
current_layer = ''
layer_to_time = {}
for line in f:
    line = line.strip()
    if 'BATCH:' in line:
        total_num_iters += 1
    elif 'Time Elapsed' in line:
        layer_to_time[current_layer].append(float((line.split())[3]))
    elif 'REPORT' in line:
        if line not in layer_to_time.keys():
            layer_to_time[line] = []
        current_layer = line

print 'Detailed Profiling Report'
print 'Average over ' + str(total_num_iters) + ' iterations'

# Make a new dict which maps to the mean only
layer_to_mean_time = {}
for k in layer_to_time.keys():
    time_list = layer_to_time[k]
    total_sum = sum(time_list)
    mean = total_sum / total_num_iters
    layer_to_mean_time[k] = mean

# Now print the means sorted
sorted_by_mean = sorted(layer_to_mean_time.items(), key=operator.itemgetter(1))
for layer_mean in reversed(sorted_by_mean):
    print layer_mean[0] + "\t" + str(layer_mean[1])

f.close()
