# AlexNet FW GEMM Sizes:
#
#         d_out     b*m^2                   d_i*k^2   
# Conv 1:  96   x  774400  (m^2=3025)   x   363
# Conv 2: 256   x  186624  (m^2= 729)   x   2400
# Conv 3: 384   x   43264  (m^2= 169)   x   2304
# Conv 4: 384   x   43264  (m^2= 169)   x   3456
# Conv 5: 256   x   43264  (m^2= 169)   x   3456

# ./sgemm_batching  batch_size  b_p  nThreads  m^2  d_out  d_in*k^2

# AlexNet Conv 2 (largest GEMM)
./sgemm_batching  256  1    8  729  256  2400
./sgemm_batching  256  2    8  729  256  2400
./sgemm_batching  256  4    8  729  256  2400
./sgemm_batching  256  8    8  729  256  2400
./sgemm_batching  256  16   8  729  256  2400
./sgemm_batching  256  32   8  729  256  2400
./sgemm_batching  256  64   8  729  256  2400
./sgemm_batching  256  128  8  729  256  2400
./sgemm_batching  256  256  8  729  256  2400

# Run other #threads at 256
./sgemm_batching  256  256 16  729  256  2400
./sgemm_batching  256  256  4  729  256  2400
./sgemm_batching  256  256  2  729  256  2400
./sgemm_batching  256  256  1  729  256  2400

