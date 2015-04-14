import sys
if len(sys.argv) != 2:
    print 'Usage: >>> python process_perf_test.py filename'
    sys.exit(0)

f = open(sys.argv[1])
test_to_metric_table = {}
current_test = ''
current_metric = ''
for line in f:
    line = line.strip()
    if not line:
        continue
    if '[ RUN      ]' in line:
        current_test = ( line.split() )[-1]
        if current_test not in test_to_metric_table.keys():
            test_to_metric_table[current_test] = {}
    elif 'report_' in line:
        current_metric = line
    elif 'Time Elapsed' in line:
        test_to_metric_table[current_test][current_metric] = (line.split())[3]
        
for i,test in enumerate(sorted(list(test_to_metric_table.keys()))):
    print test
    for metric in test_to_metric_table[test]:
       print "  " + metric[7:] + "\t" + test_to_metric_table[test][metric]
