import numpy as np
import os

# -----------------------------------------------------------------------------------------------
# Read in proto
# -----------------------------------------------------------------------------------------------
solver    = 'inputs/caffenet_solver_8_2gpu.prototxt'
train_val = 'inputs/caffenet_train_val_8_2gpu.prototxt'
cct_path = '/afs/cs.stanford.edu/u/shadjis/Raiders1/dcct/CaffeConTroll/caffe-ct'


# -----------------------------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------------------------
# LR_list = [0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.0512, 0.1024, 0.2048, 0.4096, 0.8192, 1.6384]
LR_list = [0.0001]
# M_list = [0.0, 0.3, 0.6, 0.9] # No staleness
M_list = [0.9]
# B_list = [1,4,16,64,256,1024]
B_list = [1]

epochs_phase1 = 10
num_examples = 10400


# -----------------------------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------------------------
def run_experiment(solver, train_val, lr, m, b, num_iter):

    # Make the output directory
    output_dir = 'outputs/experiment_LR' + str(lr) + '_M' + str(m) + '_B' + str(b)
    os.system('mkdir -p ' + output_dir)
    
    solver_out_name = output_dir + '/solver.prototxt'
    train_val_out_name = output_dir + '/train_val.prototxt'
    
    # Open the solver
    solver_out = open(solver_out_name, 'w')
    solver_in = open(solver)
    for line in solver_in:
        if 'momentum:' in line:
            solver_out.write('momentum: ' + str(m) + "\n")
        elif 'base_lr:' in line:
            solver_out.write('base_lr: ' + str(lr) + "\n")
        elif 'max_iter:' in line:
            solver_out.write('max_iter: ' + str(num_iter) + "\n")
        elif 'net:' in line:
            solver_out.write('net: \"' + train_val_out_name + "\"\n")
        else:
            solver_out.write(line)
    solver_in.close()
    solver_out.close()
    
    # Open the train_val
    train_val_out = open(train_val_out_name, 'w')
    train_val_in = open(train_val)
    for line in train_val_in:
        if '    batch_size:' in line:
            train_val_out.write('    batch_size: ' + str(b) + "\n")
        else:
            train_val_out.write(line)
    train_val_in.close()
    train_val_out.close()
    
    # Run CcT
    log_out = output_dir + '/log.out'
    cct_cmd = cct_path + ' train ' + solver_out_name + ' &> ' + log_out
    print cct_cmd
    os.system(cct_cmd)
    
    # Parse that log
    # Read the 
    #logfile = open(log_out)
    #for line in logfile
    #    
    #logfile.close()
    

# -----------------------------------------------------------------------------------------------
# Run experiments
# -----------------------------------------------------------------------------------------------
for b in B_list:
    print '--------------------------------------------------------------------------------'
    print 'Beginning batch size ' + str(b)
    print '--------------------------------------------------------------------------------'
    
    # First round: Run 1 min each LR
    for lr in LR_list:
        for m in M_list:
            final_acc = run_experiment(solver, train_val, lr, m, b, num_examples / b * epochs_phase1)
                
    # Second round: Pick best few from above and run longer
    
    
    
    
