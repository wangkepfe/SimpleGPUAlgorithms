
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>

#define GpuErrorCheck(ans) { GpuAssert((ans), __FILE__, __LINE__); }
inline void GpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#if 0

__global__ void prefixScan(int* num, int n)
{
	int i = threadIdx.x;

	// Bottom-up
	//
	// thread id
	//                 1                        0
	// array
	//     0           1             2          3
	//     0          0+1            2         2+3
	//     0          0+1            2       0+1+2+3

	for (int step = 1; step < n; step *= 2)
	{
		__syncthreads();

		if (i % step == 0)
		{
			int rightIdx = n - 1 - 2 * i;
			int leftIdx = rightIdx - step;

			num[rightIdx] += num[leftIdx];
		}
	}

	// Top-down
	//
	// thread id
	//                 1                        0
	// array
	//     0          0+1            2         "0"
	//     0          "0"            2         0+1
	//    "0"          0            0+1       0+1+2

	if (i == 0)
	{
		num[n - 1] = 0;
	}

	for (int step = n >> 1; step >= 1; step >>= 1)
	{
		__syncthreads();

		if (i % step == 0)
		{
			int rightIdx = n - 1 - 2 * i;
			int leftIdx = rightIdx - step;

			int left = num[leftIdx];
			int right = num[rightIdx];

			num[leftIdx] = right;
			num[rightIdx] = left + right;
		}
	}
}

extern __shared__ int lds[];

__global__ void prefixScanLds(int* num, int n)
{
	int i = threadIdx.x;

	lds[i * 2] = num[i * 2];
	lds[i * 2 + 1] = num[i * 2 + 1];

	for (int step = 1; step < n; step *= 2)
	{
		__syncthreads();

		if (i % step == 0)
		{
			int rightIdx = n - 1 - 2 * i;
			int leftIdx = rightIdx - step;

			lds[rightIdx] += lds[leftIdx];
		}
	}

	if (i == 0)
	{
		lds[n - 1] = 0;
	}

	for (int step = n >> 1; step >= 1; step >>= 1)
	{
		__syncthreads();

		if (i % step == 0)
		{
			int rightIdx = n - 1 - 2 * i;
			int leftIdx = rightIdx - step;

			int left = lds[leftIdx];
			int right = lds[rightIdx];

			lds[leftIdx] = right;
			lds[rightIdx] = left + right;
		}
	}

	__syncthreads();

	num[i * 2] = lds[i * 2];
	num[i * 2 + 1] = lds[i * 2 + 1];
}

__global__ void prefixScanV3(int* num, int n)
{
	int i = threadIdx.x;

	lds[i * 2] = num[i * 2];
	lds[i * 2 + 1] = num[i * 2 + 1];

	__syncthreads();

	// Bottom-up
	//
	// array
	//     0           1           2          3         4          5           6          7
	// thread id
	//                 3                      2                    1                      0
	// array
	//     0          0+1          2         2+3        4         4+5          6         6+7
	// thread id
	//                                        1                                           0
	// array
	//     0          0+1          2       0+1+2+3      4         4+5          6       4+5+6+7
	// thread id
	//                                                                                    0
	// array
	//     0          0+1          2       0+1+2+3      4         4+5          6   0+1+2+3+4+5+6+7
	for (int step = 1; step < n; step *= 2)
	{
		if (i < n / 2 / step)
		{
			int rightIdx = n - 1 - 2 * i * step;
			int leftIdx = rightIdx - step;

			lds[rightIdx] += lds[leftIdx];
		}
		__syncthreads();
	}

	if (i == 0)
	{
		lds[n - 1] = 0;
	}

	__syncthreads();

	for (int step = n >> 1; step >= 1; step >>= 1)
	{
		if (i < n / 2 / step)
		{
			int rightIdx = n - 1 - 2 * i * step;
			int leftIdx = rightIdx - step;

			int left = lds[leftIdx];
			int right = lds[rightIdx];

			lds[leftIdx] = right;
			lds[rightIdx] = left + right;
		}
		__syncthreads();
	}

	num[i * 2] = lds[i * 2];
	num[i * 2 + 1] = lds[i * 2 + 1];
}

template<int n>
__forceinline__ __device__ void scan()
{
	int i = threadIdx.x;
	int step;

#pragma unroll
	for (step = 1; step < 32; step *= 2)
	{
		if (i < n / 2 / step)
		{
			int rightIdx = n - 1 - 2 * i * step;
			int leftIdx = rightIdx - step;

			lds[rightIdx] += lds[leftIdx];
		}
	}

#pragma unroll
	for (; step < n; step *= 2)
	{
		if (i < n / 2 / step)
		{
			int rightIdx = n - 1 - 2 * i * step;
			int leftIdx = rightIdx - step;

			lds[rightIdx] += lds[leftIdx];
		}
		__syncthreads();
	}

	if (i == 0)
	{
		lds[n - 1] = 0;
	}

	__syncthreads();

#pragma unroll
	for (step = n >> 1; step >= 32; step >>= 1)
	{
		if (i < n / 2 / step)
		{
			int rightIdx = n - 1 - 2 * i * step;
			int leftIdx = rightIdx - step;

			int left = lds[leftIdx];
			int right = lds[rightIdx];

			lds[leftIdx] = right;
			lds[rightIdx] = left + right;
		}
		__syncthreads();
	}

#pragma unroll
	for (step = 16; step >= 1; step >>= 1)
	{
		if (i < n / 2 / step)
		{
			int rightIdx = n - 1 - 2 * i * step;
			int leftIdx = rightIdx - step;

			int left = lds[leftIdx];
			int right = lds[rightIdx];

			lds[leftIdx] = right;
			lds[rightIdx] = left + right;
		}
	}
}

template<int n, int m>
__global__ void prefixScanV4(int* num, int count)
{
	int i = threadIdx.x;

	int idx1 = blockIdx.x * n + i * 2;
	int idx2 = blockIdx.x * n + i * 2 + 1;

	lds[i * 2] = idx1 < count ? num[idx1] : 0;
	lds[i * 2 + 1] = idx2 < count ? num[idx2] : 0;

	__syncthreads();

	scan<n>();

	__syncthreads();

	if (idx1 < count) num[idx1] = lds[i * 2];
	if (idx2 < count) num[idx2] = lds[i * 2 + 1];

	__syncthreads();

	if (i < m / 2)
	{
		int idx3 = (i * 2) * n + 2047;
		int idx4 = (i * 2 + 1) * n + 2047;

		lds[i * 2] = idx3 < count ? num[idx3] : 0;
		lds[i * 2 + 1] = idx4 < count ? num[idx4] : 0;
	}

	__syncthreads();

	scan<m>();

	__syncthreads();

	if (idx1 < count) num[idx1] += lds[blockIdx.x];
	if (idx2 < count) num[idx2] += lds[blockIdx.x];
}

template<int n>
__global__ void prefixScanV5_block(volatile int* num, volatile int* num2, int count)
{
	extern __shared__ int lds[];
	int i = threadIdx.x;
	int idx1 = blockIdx.x * n + i * 2;
	int idx2 = blockIdx.x * n + i * 2 + 1;
	lds[i * 2] = idx1 < count ? num[idx1] : 0;
	lds[i * 2 + 1] = idx2 < count ? num[idx2] : 0;
	__syncthreads();
	int step;
	//#pragma unroll
	for (step = 1; step < n; step *= 2)
	{
		__syncthreads();
		if (i < (n / 2) / step)
		{
			int rightIdx = n - 1 - 2 * i * step;
			int leftIdx = rightIdx - step;

			lds[rightIdx] += lds[leftIdx];
		}
		__syncthreads();
	}
	if (i == blockDim.x - 1)
	{
		lds[n - 1] = 0;
	}
	__syncthreads();
	// #pragma unroll
	for (step = n >> 1; step > 1; step >>= 1)
	{
		__syncthreads();
		if (i < (n / 2) / step)
		{
			int rightIdx = n - 1 - 2 * i * step;
			int leftIdx = rightIdx - step;

			int left = lds[leftIdx];
			int right = lds[rightIdx];

			lds[leftIdx] = right;
			lds[rightIdx] = left + right;
		}
		__syncthreads();
	}
	if (i == blockDim.x - 1)
	{
		num2[blockIdx.x] = num[idx2] + lds[i * 2 + 1];
	}
	if (idx1 < count) num[idx1] = lds[i * 2];
	if (idx2 < count) num[idx2] = lds[i * 2 + 1];
}

template<int n>
__global__ void prefixScanV5_grid(volatile int* num, int count)
{
	extern __shared__ int lds[];
	int i = threadIdx.x;
	int idx1 = i * 2;
	int idx2 = i * 2 + 1;
	lds[idx1] = idx1 < count ? num[idx1] : 0;
	lds[idx2] = idx2 < count ? num[idx2] : 0;
	__syncthreads();
	int step;
#pragma unroll
	for (step = 1; step < 32; step *= 2)
	{
		if (i < (n / 2) / step)
		{
			int rightIdx = n - 1 - 2 * i * step;
			int leftIdx = rightIdx - step;

			lds[rightIdx] += lds[leftIdx];
		}
	}
#pragma unroll
	for (step = 32; step < n; step *= 2)
	{
		if (i < (n / 2) / step)
		{
			int rightIdx = n - 1 - 2 * i * step;
			int leftIdx = rightIdx - step;

			lds[rightIdx] += lds[leftIdx];
		}
		__syncthreads();
	}
	if (i == 0)
	{
		lds[n - 1] = 0;
	}
	__syncthreads();
#pragma unroll
	for (step = n >> 1; step >= 32; step >>= 1)
	{
		if (i < (n / 2) / step)
		{
			int rightIdx = n - 1 - 2 * i * step;
			int leftIdx = rightIdx - step;

			int left = lds[leftIdx];
			int right = lds[rightIdx];

			lds[leftIdx] = right;
			lds[rightIdx] = left + right;
		}
		__syncthreads();
	}
#pragma unroll
	for (step = 16; step >= 1; step >>= 1)
	{
		if (i < (n / 2) / step)
		{
			int rightIdx = n - 1 - 2 * i * step;
			int leftIdx = rightIdx - step;

			int left = lds[leftIdx];
			int right = lds[rightIdx];

			lds[leftIdx] = right;
			lds[rightIdx] = left + right;
		}
	}
	__syncthreads();
	if (idx1 < count) num[idx1] = lds[idx1];
	if (idx2 < count) num[idx2] = lds[idx2];
}

template<int n>
__global__ void prefixScanV5_add(volatile int* num, volatile int* num2, int count)
{
	int i = threadIdx.x;
	int j = blockIdx.x;

	int idx1 = j * n + i * 2;
	int idx2 = j * n + i * 2 + 1;

	int blockScanRes = num2[j];
	if (idx1 < count) num[idx1] += blockScanRes;
	if (idx2 < count) num[idx2] += blockScanRes;
}
#endif

int getTwoExpPaddedSize(int n)
{
	int res = 1;
	while (n > res) res <<= 1;
	return res;
}

int main()
{
	srand(time(NULL));

	const int numCount = 16; // max 4,194,304
	const int numSize = numCount * sizeof(int);

	int* num_host = new int[numCount];

	int sum = 0;

	std::cout << sum << ",";

	for (int i = 0; i < numCount; ++i)
	{
		num_host[i] = rand() % 8;
		sum += num_host[i];
		if (i != numCount - 1)
			std::cout << sum << ",";
	}

	sum -= num_host[numCount - 1];

	std::cout << "\n\n\n";

	std::cout << sum << "\n";

	std::cout << "\n\n\n";

	int* num_device;
	int* gridNum_d;

	cudaMalloc((void**)& num_device, numSize);
	cudaMemcpy(num_device, num_host, numSize, cudaMemcpyHostToDevice);

	int gridDim = (numCount + 2047) / 2048;

	int* gridNum_h = new int[gridDim];
	cudaMalloc((void**)& gridNum_d, gridDim * sizeof(int));
	cudaMemset(gridNum_d, 0, gridDim * sizeof(int));

	int paddedGridDim = getTwoExpPaddedSize(gridDim);

#if 0
	prefixScan << < dim3(1, 1, 1), dim3(padCount / 2, 1, 1) >> > (num_device, padCount);
	prefixScanLds << < dim3(1, 1, 1), dim3(padCount / 2, 1, 1), padSize >> > (num_device, padCount);
	prefixScanV3 << < dim3(1, 1, 1), dim3(padCount / 2, 1, 1), padSize >> > (num_device, padCount);

	switch (paddedGridDim)
	{
	case 1: prefixScanV4 <2048, 1> << < dim3(gridDim, 1, 1), dim3(1024, 1, 1), 2048 * sizeof(int) >> > (num_device, numCount); break;
	case 2: prefixScanV4 <2048, 2> << < dim3(gridDim, 1, 1), dim3(1024, 1, 1), 2048 * sizeof(int) >> > (num_device, numCount); break;
	case 4: prefixScanV4 <2048, 4> << < dim3(gridDim, 1, 1), dim3(1024, 1, 1), 2048 * sizeof(int) >> > (num_device, numCount); break;
	case 8: prefixScanV4 <2048, 8> << < dim3(gridDim, 1, 1), dim3(1024, 1, 1), 2048 * sizeof(int) >> > (num_device, numCount); break;
	case 16: prefixScanV4 <2048, 16> << < dim3(gridDim, 1, 1), dim3(1024, 1, 1), 2048 * sizeof(int) >> > (num_device, numCount); break;
	case 32: prefixScanV4 <2048, 32> << < dim3(gridDim, 1, 1), dim3(1024, 1, 1), 2048 * sizeof(int) >> > (num_device, numCount); break;
	case 64: prefixScanV4 <2048, 64> << < dim3(gridDim, 1, 1), dim3(1024, 1, 1), 2048 * sizeof(int) >> > (num_device, numCount); break;
	case 128: prefixScanV4 <2048, 128> << < dim3(gridDim, 1, 1), dim3(1024, 1, 1), 2048 * sizeof(int) >> > (num_device, numCount); break;
	case 256: prefixScanV4 <2048, 256> << < dim3(gridDim, 1, 1), dim3(1024, 1, 1), 2048 * sizeof(int) >> > (num_device, numCount); break;
	case 512: prefixScanV4 <2048, 512> << < dim3(gridDim, 1, 1), dim3(1024, 1, 1), 2048 * sizeof(int) >> > (num_device, numCount); break;
	case 1024: prefixScanV4 <2048, 1024> << < dim3(gridDim, 1, 1), dim3(1024, 1, 1), 2048 * sizeof(int) >> > (num_device, numCount); break;
	case 2048: prefixScanV4 <2048, 2048> << < dim3(gridDim, 1, 1), dim3(1024, 1, 1), 2048 * sizeof(int) >> > (num_device, numCount); break;
	}
#endif

	prefixScanV5_block <16> << < dim3(gridDim, 1, 1), dim3(8, 1, 1), 16 * sizeof(int) >> > (num_device, gridNum_d, numCount);
	cudaDeviceSynchronize();

	GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());

	// switch (paddedGridDim)
	// {
	// 	case 1: prefixScanV5_grid <1> << < dim3(1, 1, 1), dim3(1, 1, 1), 1 * sizeof(int) >> > (gridNum_d, gridDim); break;
	// 	case 2: prefixScanV5_grid <2> << < dim3(1, 1, 1), dim3(1, 1, 1), 2 * sizeof(int) >> > (gridNum_d, gridDim); break;
	// 	case 4: prefixScanV5_grid <4> << < dim3(1, 1, 1), dim3(2, 1, 1), 4 * sizeof(int) >> > (gridNum_d, gridDim); break;
	// 	case 8: prefixScanV5_grid <8> << < dim3(1, 1, 1), dim3(4, 1, 1), 8 * sizeof(int) >> > (gridNum_d, gridDim); break;
	// 	case 16: prefixScanV5_grid <16> << < dim3(1, 1, 1), dim3(8, 1, 1), 16 * sizeof(int) >> > (gridNum_d, gridDim); break;
	// 	case 32: prefixScanV5_grid <32> << < dim3(1, 1, 1), dim3(16, 1, 1), 32 * sizeof(int) >> > (gridNum_d, gridDim); break;
	// 	case 64: prefixScanV5_grid <64> << < dim3(1, 1, 1), dim3(32, 1, 1), 64 * sizeof(int) >> > (gridNum_d, gridDim); break;
	// 	case 128: prefixScanV5_grid <128> << < dim3(1, 1, 1), dim3(64, 1, 1), 128 * sizeof(int) >> > (gridNum_d, gridDim); break;
	// 	case 256: prefixScanV5_grid <256> << < dim3(1, 1, 1), dim3(128, 1, 1), 256 * sizeof(int) >> > (gridNum_d, gridDim); break;
	// 	case 512: prefixScanV5_grid <512> << < dim3(1, 1, 1), dim3(256, 1, 1), 512 * sizeof(int) >> > (gridNum_d, gridDim); break;
	// 	case 1024: prefixScanV5_grid <1024> << < dim3(1, 1, 1), dim3(512, 1, 1), 1024 * sizeof(int) >> > (gridNum_d, gridDim); break;
	// 	case 2048: prefixScanV5_grid <2048> << < dim3(1, 1, 1), dim3(1024, 1, 1), 2048 * sizeof(int) >> > (gridNum_d, gridDim); break;
	// }

	// GpuErrorCheck(cudaDeviceSynchronize());
	// GpuErrorCheck(cudaPeekAtLastError());

	// prefixScanV5_add<2048> << < dim3(gridDim, 1, 1), dim3(1024, 1, 1) >> > (num_device, gridNum_d, numCount);

	// GpuErrorCheck(cudaDeviceSynchronize());
	// GpuErrorCheck(cudaPeekAtLastError());

	cudaMemcpy(num_host, num_device, numSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(gridNum_h, gridNum_d, gridDim * sizeof(int), cudaMemcpyDeviceToHost);

	std::cout << "\n\n\n";

	for (int i = 0; i < gridDim; ++i)
	{
		std::cout << gridNum_h[i] << ",";
	}

	std::cout << "\n\n\n";

	for (int i = 0; i < numCount; ++i)
	{
		std::cout << num_host[i] << ",";
	}

	std::cout << "\n\n\n";

	std::cout << num_host[numCount - 1] << "\n";

	delete num_host;
	delete gridNum_h;

	cudaFree(gridNum_d);
	cudaFree(num_device);

	return 0;
}
