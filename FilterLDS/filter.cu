
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <iomanip>

#define GpuErrorCheck(ans) { GpuAssert((ans), __FILE__, __LINE__); }
inline void GpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

int DivRound(int n, int m) { return (n + m - 1) / m; }

__device__ int max3(int a, int b, int c) { return max(a, max(b, c)); }

__device__ int LoadBuffer(int* data, int w, int h, int x, int y)
{
	x = min(max(x, 0), w - 1);
	y = min(max(y, 0), h - 1);
	return data[x + y * w];
}

__global__ void MaxFilterNaive(int* data, int w, int h)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int v[9];
	for (int i = 0; i <= 2; ++i)
	{
		for (int j = 0; j <= 2; ++j)
		{
			v[j + 3 * i] = LoadBuffer(data, w, h, x + i - 1, y + j - 1);
		}
	}

	int result = 0;
	for (int i = 0; i <= 2; ++i)
	{
		for (int j = 0; j <= 2; ++j)
		{
			result = max(result, v[j + 3 * i]);
		}
	}

	data[x + y * w] = result;
}

__global__ void MaxFilterLds(int* data, int w, int h)
{
	__shared__ int lds[100];

	int x = threadIdx.x + blockIdx.x * 8;
	int y = threadIdx.y + blockIdx.y * 8;

	int id = (threadIdx.x + threadIdx.y * 8);

	int x1 = blockIdx.x * 8 - 1 + id % 10;
	int y1 = blockIdx.y * 8 - 1 + id / 10;

	int x2 = blockIdx.x * 8 - 1 + (id + 64) % 10;
	int y2 = blockIdx.y * 8 - 1 + (id + 64) / 10;

	int v1 = LoadBuffer(data, w, h, x1, y1);
	int v2 = LoadBuffer(data, w, h, x2, y2);

	lds[id] = v1;
	lds[id + 64] = v2;

	__syncthreads();

	int result = 0;
	for (int i = 0; i <= 2; ++i)
	{
		for (int j = 0; j <= 2; ++j)
		{
			result = max(result, lds[threadIdx.x + j + (threadIdx.y + i) * 10]);
		}
	}

	data[x + y * w] = result;
}

__global__ void MaxFilterDLP(int* data, int w, int h)
{
	int laneId = threadIdx.x + threadIdx.y * blockDim.x;

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int v_1, v_2;

	int v1 = LoadBuffer(data, w, h, x - 1, y - 1); // row [0:3] = [-1:2] left
	int v2 = LoadBuffer(data, w, h, x + 1, y - 1); // row [0:3] = [-1:2] right
	int v3 = LoadBuffer(data, w, h, (laneId < 16) ? (x - 1) : (x + 1), (laneId < 16) ? (y + 3) : (y + 1)); // row [0:1] = [3:4] left, row [2:3] = [3:4] right

	v_1 = __shfl_sync(0xffffffff, v1, laneId + 1);
	v_2 = __shfl_sync(0xffffffff, v2, laneId - 1);
	int v4 = (threadIdx.x < 7) ? v_1 : v_2; // row [0:3] = [-1:2] middle

	v_1 = __shfl_sync(0xffffffff, v3, laneId + 1);
	v_2 = __shfl_sync(0xffffffff, v3, (laneId ^ 0x10) - 1);
	int v5 = (threadIdx.x < 7) ? v_1 : v_2; // row [0:1] = [3:4] middle
	int v6 = __shfl_sync(0xffffffff, v3, laneId ^ 0x10); // row [0:1] = [3:4] right

	v1 = max3(v1, v2, v4); // row [0:3] = [-1:2] horizontal max ;  = [0:3] top
	v2 = max3(v3, v5, v6); // row [0:1] = [3:4] horizontal max ;

	v_1 = __shfl_sync(0xffffffff, v1, laneId + 8);
	v_2 = __shfl_sync(0xffffffff, v2, laneId & 0x7);
	v3 = (threadIdx.y < 3) ? v_1 : v_2; // row [0:3] = [0:3] middle

	v_1 = __shfl_sync(0xffffffff, v1, laneId ^ 0x10);
	v_2 = __shfl_sync(0xffffffff, v2, laneId ^ 0x10);
	v4 = (threadIdx.y < 2) ? v_1 : v_2; // row [0:3] = [0:3] bottom

	v1 = max3(v1, v3, v4);

	data[x + y * w] = v1;
}

int main()
{
	srand(time(NULL));

	// create cpu buffer
	int w = 8 * 1000;
	int h = 8 * 1000;
	int numCount = h * w;
	int* h_data = new int[numCount];
	for (int i = 0; i < numCount; ++i) { h_data[i] = rand() % 16; }

	// print original
	//for (int i = 0; i < h; ++i)
	//{
	//	for (int j = 0; j < w; ++j)
	//	{
	//		std::cout << std::setw(3) << std::left << h_data[i * w + j] << ",";
	//		if (j % 8 == 7) std::cout << "   ,";
	//	}
	//	std::cout << "\n";
	//	if (i % 8 == 7) std::cout << "\n";
	//}
	//std::cout << "\n\n";

	// create gpu buffer and copy to gpu
	int* d_data1;
	int* d_data2;
	int* d_data3;

	GpuErrorCheck(cudaMalloc((void**)& d_data1, numCount * sizeof(int)));
	GpuErrorCheck(cudaMemcpy(d_data1, h_data, numCount * sizeof(int), cudaMemcpyHostToDevice));

	GpuErrorCheck(cudaMalloc((void**)& d_data2, numCount * sizeof(int)));
	GpuErrorCheck(cudaMemcpy(d_data2, h_data, numCount * sizeof(int), cudaMemcpyHostToDevice));

	GpuErrorCheck(cudaMalloc((void**)& d_data3, numCount * sizeof(int)));
	GpuErrorCheck(cudaMemcpy(d_data3, h_data, numCount * sizeof(int), cudaMemcpyHostToDevice));

	// dispatch
	MaxFilterNaive << < dim3(DivRound(w, 8), DivRound(h, 8), 1), dim3(8, 8, 1) >> > (d_data1, w, h);
	MaxFilterLds << < dim3(DivRound(w, 8), DivRound(h, 8), 1), dim3(8, 8, 1) >> > (d_data2, w, h);
	MaxFilterDLP << < dim3(DivRound(w, 8), DivRound(h, 4), 1), dim3(8, 4, 1) >> > (d_data3, w, h);

	GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());

	// copy to cpu
	GpuErrorCheck(cudaMemcpy(h_data, d_data1, numCount * sizeof(int), cudaMemcpyDeviceToHost));

	// print processed
	//for (int i = 0; i < h; ++i)
	//{
	//	for (int j = 0; j < w; ++j)
	//	{
	//		std::cout << std::setw(3) << std::left << h_data[i * w + j] << ",";
	//		if (j % 8 == 7) std::cout << "   ,";
	//	}
	//	std::cout << "\n";
	//	if (i % 8 == 7) std::cout << "\n";
	//}

	delete h_data;
	cudaFree(d_data1);
	cudaFree(d_data2);
	cudaFree(d_data3);
	return 0;
}