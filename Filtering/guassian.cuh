
#pragma once
#include "common.cuh"

__constant__ float cGaussian[25];

__device__ Float3 Load(Float3* in, Int2 pos, Int2 size)
{
    if (pos.x < 0) pos.x = 0;
    if (pos.y < 0) pos.y = 0;
    if (pos.x > size.x - 1) pos.x = size.x - 1;
    if (pos.y > size.y - 1) pos.y = size.y - 1;
    return in[pos.y * size.x + pos.x];
}

__device__ void Store(Float3* out, Float3 val, Int2 pos, Int2 size)
{
    if (pos.x < 0) pos.x = 0;
    if (pos.y < 0) pos.y = 0;
    if (pos.x > size.x - 1) pos.x = size.x - 1;
    if (pos.y > size.y - 1) pos.y = size.y - 1;
    out[pos.y * size.x + pos.x] = val;
}

__device__ float GetGaussian(int x, int y)
{
    return cGaussian[(y + 2) * 5 + (x + 2)];
}

__global__ void Copy( Float3* in,  Float3* out, Int2 size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	Int2 idx(x, y);
	Float3 val = Load(in, idx, size);
	Store(out, val, idx, size);
}

__global__ void GaussianFilter(Float3* in,  Float3* out, Int2 size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
    Float3 colorSum = 0;
    float weightSum = 0;
    for (int i = x - 2; i <= x + 2; ++i)
    {
        for (int j = y - 2; j <= y + 2; ++j)
        {
            Float3 color = Load(in, Int2(i, j), size);
            float weight = GetGaussian(i - x, j - y);
            colorSum += color * weight;
            weightSum += weight;
        }
    }
    Float3 finalColor = colorSum / weightSum;
	Store(out, finalColor, Int2(x, y), size);
}

float GaussianIsotropic2D(float x, float y, float sigma)
{
    return expf(- (x * x + y * y) / (2 * sigma * sigma) ) / (2 * PI * sigma * sigma);
}

void CalculateGaussianKernel(float sigma, int radius = 2)
{
    int size = radius * 2 + 1;
    const int step = 100;
    int kernelSize = size * size;
    int sampleDimSize = size * step;
    int sampleCount = sampleDimSize * sampleDimSize;
    float *sampleData = new float[sampleCount];
    for (int i = 0; i < sampleCount; ++i)
    {
        int xi = i % sampleDimSize;
        int yi = i / sampleDimSize;
        float x = (float)xi / (float)step;
        float y = (float)yi / (float)step;
        float offset = (float)size / 2;
        x -= offset;
        y -= offset;
        sampleData[i] = GaussianIsotropic2D(x, y, sigma);
    }
    float* fGaussian = new float[kernelSize];
    for (int i = 0; i < kernelSize; ++i)
    {
        int xi = i % size;
        int yi = i / size;
        float valSum = 0;
        for (int x = xi * step; x < (xi + 1) * step; ++x)
        {
            for (int y = yi * step; y < (yi + 1) * step; ++y)
            {
                valSum += sampleData[y * sampleDimSize + x];
            }
        }
        fGaussian[i] = valSum / (step * step);
    }
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            std::cout << fGaussian[i + j * size] << ", ";
        }
        std::cout << std::endl;
    }
	GpuErrorCheck(cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float) * kernelSize));
    delete fGaussian;
    delete sampleData;
}