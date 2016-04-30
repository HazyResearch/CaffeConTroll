#include "cblas.h"
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

const int NUMBER = 10;
void run_batch_experiment(const int b_p, const int batch_size, const int n, const int m, const int r) {
  
  struct timeval start, end;
  float total_time = 0.;
  int ngemm = 0;
  bool first = true;
  
  // std::cout << "Experiment for convolution batch size b_p = " << b_p << ":" << std::endl;
    
  for(int i = 0; i < NUMBER; ++i) { 
    for (int b_i=0; b_i<batch_size; b_i += b_p) {
   
      float * A = new float [m*r];
      float * B = new float [r*n*b_p];
      float * C = new float [m*n*b_p];
      
      // Generate random data
      srand((unsigned int)0x100);
      // std::cout << "Generating Matrix Data... ";
      // std::cout.flush(); 
      for(int i = 0; i < m; ++i) {
        for(int j = 0; j < r; ++j) {
          A[i*r+j] = float(rand()%100) / 100.0;
        }
      }
      for(int i = 0; i < r; ++i) {
        for(int j = 0; j < n*b_p; ++j) {
          B[i*n*b_p+j] = float(rand()%100) / 100.0;
        }
      }
      for(int i = 0; i < m; ++i){
        for(int j = 0; j < n*b_p; ++j) {
          C[i*n*b_p+j] = 0.0;
        }
      }
      if (first) {
        std::cout << "Memory usage = " << float((m*r + r*n*b_p + m*n*b_p)*4)/(1024*1024*1024) << " GB" << std::endl;
        first = false;
      }
      // std::cout << "Running GEMM...";
      // std::cout.flush(); 
      
      gettimeofday(&start, NULL);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  m,n*b_p,r, 1.0,A,r,B,n*b_p,0.0, C, n*b_p);
      gettimeofday(&end, NULL);
      long seconds  = end.tv_sec  - start.tv_sec;
      long useconds = end.tv_usec - start.tv_usec;  
      float mtime = ((seconds) * 1000 + useconds/1000.0);
      total_time += mtime;
      ngemm += 1;
      
      delete A;
      delete B;
      delete C;
    }
  }
  printf("Total # GEMM calls = %d\n", ngemm / NUMBER);
  printf("Elapsed time of GEMM calls: %f milliseconds GFlops=%f\n", total_time, ((float) NUMBER*2*m*n*batch_size*r)/(total_time*1e6));
}

int main (int argc, char *argv[]) {
  if (argc != 7) {
    std::cout << "Usage:  ./sgemm_batching  batch_size  b_p  nThreads  m^2  d_out  d_in*k^2" << std::endl;
    std::cout << "" << std::endl;
    exit(0);
  }
  int batch_size = atoi(argv[1]);
  int b_p = atoi(argv[2]);
  int nThreads = atoi(argv[3]);
  int n = atoi(argv[4]);
  int m = atoi(argv[5]);
  int r = atoi(argv[6]);
  std::cout << "-----------------------------------------------------------------------" << std::endl;
  std::cout << "Running with batch size " << batch_size << ", b_p = " << b_p << ", nThreads = " << nThreads << std::endl;
  std::cout << "Conv Parameters: m^2 = " << n << ", d_out = " << m << ", d_in*k^2 = " << r << std::endl;
  std::cout << "-----------------------------------------------------------------------" << std::endl;
  openblas_set_num_threads(nThreads);
  run_batch_experiment(b_p, batch_size, n, m, r);
  std::cout << std::endl;
  return 0;
}
