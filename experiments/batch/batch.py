import numpy as np
import os

# -----------------------------------------------------------------------------------------------
# Read in proto
# -----------------------------------------------------------------------------------------------
solver    = 'inputs/caffenet_solver_8_4gpu.prototxt'
train_val = 'inputs/caffenet_train_val_8_4gpu.prototxt'
cct_path = '/home/software/CaffeConTroll/caffe-ct'


# -----------------------------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------------------------
LR_list = [0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024, 0.2048, 0.4096, 0.8192, 1.6384]
# M_list = [0.0, 0.3, 0.6, 0.9] # No staleness
M_list = [0.9]
B_list = [4,16,64,256]

# Do in intervals of highest batch size so it is fair / equal
biggest_batch = max(B_list)
phase1_num_big_batch_iter = 40
phase1_running_avg = 8
phase2_num_big_batch_iter = 1200


# -----------------------------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------------------------
def run_experiment(solver, train_val, lr, m, b, num_iter, display, experiment_name):

    # Make the output directory
    output_dir = 'outputs/B' + str(b) + '/' + experiment_name + '/experiment_LR' + str(lr) + '_M' + str(m)
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
            solver_out.write('base_lr: '  + str(lr) + "\n")
        elif 'max_iter:' in line:
            solver_out.write('max_iter: ' + str(num_iter) + "\n")
        elif 'display:' in line:
            solver_out.write('display: '  + str(display) + "\n")
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
    cct_cmd = cct_path + ' train ' + solver_out_name + ' > ' + log_out + ' 2>&1'
    print cct_cmd
    os.system(cct_cmd)
    
    # Parse that log
    logfile = open(log_out)
    acc_lines = []
    for line in logfile:
        if 'PARSE' in line:
            # PARSE  100  0.1278  0.878
            acc_lines.append(float(line.strip().split()[-1]))
    logfile.close()
    
    # Return the accuracies each iter
    return acc_lines
    

# -----------------------------------------------------------------------------------------------
# Run experiments
# -----------------------------------------------------------------------------------------------
for b in B_list:
    print '--------------------------------------------------------------------------------'
    print 'Beginning batch size ' + str(b)
    print '--------------------------------------------------------------------------------'
    
    # First round: Run 1 min each LR
    best_3 = []
    acc_to_lr_m = {}
    for lr in LR_list:
        for m in M_list:
            # The # images to process is  is the biggest batch times phase1_num_big_batch_iter
            total_num_imgs_to_process = biggest_batch*phase1_num_big_batch_iter
            # The # iter is this divided by the batch size
            num_iter = total_num_imgs_to_process/b
            # The display freq is biggest_batch / b, i.e. the total # displays will be phase1_num_big_batch_iter
            display = biggest_batch / b
            # Run experiment
            accuracies = run_experiment(solver, train_val, lr, m, b, num_iter, display, 'phase1')
            assert len(accuracies) == phase1_num_big_batch_iter
            # Note each list element is a display, i.e. it is biggest_batch images processed
            final_acc = sum(accuracies[-phase1_running_avg:])/phase1_running_avg
            print '  Final Acc = ' + str(final_acc)
            # Check if this is top 3, if so add it
            if len(best_3) < 3:
                best_3.append(final_acc)
                acc_to_lr_m[final_acc] = (lr, m)
                best_3 = sorted(best_3)
            elif final_acc > min(best_3):
                assert min(best_3) == best_3[0]
                best_3 = best_3[1:]
                best_3.append(final_acc)
                acc_to_lr_m[final_acc] = (lr, m)
                best_3 = sorted(best_3)
            print '  -> best_3 = ' + str(best_3)
                
    # Second round: Pick best 3 learning rates and run longer
    print ''
    for k in best_3:
        print 'Running ' + str(acc_to_lr_m[k])
        lr = acc_to_lr_m[k][0]
        m = acc_to_lr_m[k][1]
        # The # images to process is  is the biggest batch times phase1_num_big_batch_iter
        total_num_imgs_to_process = biggest_batch*phase2_num_big_batch_iter
        # The # iter is this divided by the batch size
        num_iter = total_num_imgs_to_process/b
        # The display freq is biggest_batch / b, i.e. the total # displays will be phase1_num_big_batch_iter
        display = biggest_batch / b
        # Run experiment
        accuracies = run_experiment(solver, train_val, lr, m, b, num_iter, display, 'phase2')
    
