#include "cblas.h"
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
// #include <unistd.h>

const long n = 1024*16;
const long m = n;
const long r = n;
// Number of times to run
// Note: should make new data for each run
const int NUMBER = 5;

int main() {
  
  struct timeval start, end;

  float * A = new float [m*r];
  float * B = new float [r*n];
  float * C = new float [m*n];
  
  // Generate random data
  srand((unsigned int)0x100);
  std::cout << "Generating Matrix Data: ";
  for(int i = 0; i < m; ++i) {
    for(int j = 0; j < r; ++j) {
      A[i*r+j] = float(rand()%100) / 100.0;
    }
  }
  for(int i = 0; i < r; ++i) {
    for(int j = 0; j < n; ++j) {
      B[i*n+j] = float(rand()%100) / 100.0;
    }
  }
  for(int i = 0; i < m; ++i){
    for(int j = 0; j < n; ++j) {
      C[i*n+j] = 0.0;
    }
  }
  std::cout << "Done." << std::endl;
  
  gettimeofday(&start, NULL);
  for(int i = 0; i < NUMBER; ++i) { 
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m,n,r, 1.0,A,r,B,n,0.0, C, n);
  }
  gettimeofday(&end, NULL);
  long seconds  = end.tv_sec  - start.tv_sec;
  long useconds = end.tv_usec - start.tv_usec;  
  float mtime = ((seconds) * 1000 + useconds/1000.0);
  printf("Elapsed time: %f milliseconds GFlops=%f\n", mtime, ((float) NUMBER*2*m*n*r)/(mtime*1e6));
  
  delete A;
  delete B;
  delete C;
  
  return 0;
}
