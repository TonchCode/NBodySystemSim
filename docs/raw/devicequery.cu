#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(float* o){ o[threadIdx.x] = rsqrtf((float)threadIdx.x + 1.0f); }
int main(){
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  printf("name=%s\ncc=%d.%d\nSMs=%d\nmaxThreadsPerSM=%d\nmaxBlocksPerSM=%d\nregsPerSM=%d\nsmemPerSM=%zu\nsmemPerBlockOptin=%zu\nL2=%d\nmemBusWidth=%d\nglobalMem=%zu\nwarpSize=%d\nasyncEngines=%d\nconcurrentKernels=%d\nunifiedAddressing=%d\n",
    p.name,p.major,p.minor,p.multiProcessorCount,p.maxThreadsPerMultiProcessor,p.maxBlocksPerMultiProcessor,p.regsPerMultiprocessor,p.sharedMemPerMultiprocessor,p.sharedMemPerBlockOptin,p.l2CacheSize,p.memoryBusWidth,p.totalGlobalMem,p.warpSize,p.asyncEngineCount,p.concurrentKernels,p.unifiedAddressing);
  int clk=0, mclk=0; cudaDeviceGetAttribute(&clk,cudaDevAttrClockRate,0); cudaDeviceGetAttribute(&mclk,cudaDevAttrMemoryClockRate,0);
  printf("smClockKHz=%d\nmemClockKHz=%d\n",clk,mclk);
  float* d; cudaMalloc(&d,32*4); k<<<1,32>>>(d); float h[32]; cudaMemcpy(h,d,128,cudaMemcpyDeviceToHost);
  printf("kernel=%s out[3]=%f\n", cudaGetErrorString(cudaGetLastError()), h[3]);
  int rt,dv; cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&dv); printf("runtime=%d driver=%d\n",rt,dv);
}
