
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "guassian.cuh"
#include "common.cuh"

float UcharToFloat(uchar a)
{
	return a / 255.f;
}

uchar FloatToUchar(float f)
{
	return uchar(clampf(f, 0, 1) * 255.f);
}

void ConvertUcharToFloat3(uchar* a, Float3* b, Int2 size)
{
	int s = size.x * size.y;
	for (int i = 0; i < s; ++i)
	{
		b[i].x = UcharToFloat(a[i * 3 + 0]);
		b[i].y = UcharToFloat(a[i * 3 + 1]);
		b[i].z = UcharToFloat(a[i * 3 + 2]);
	}
}

void ConvertFloat3ToUchar(uchar* a, Float3* b, Int2 size)
{
	int s = size.x * size.y;
	for (int i = 0; i < s; ++i)
	{
		a[i * 3 + 0] = FloatToUchar(b[i].x);
		a[i * 3 + 1] = FloatToUchar(b[i].y);
		a[i * 3 + 2] = FloatToUchar(b[i].z);
	}
}

enum class FilterType
{
	Copy,
	Gaussian,
};

void RunFilter(uchar* in, uchar* out, Int2 size, FilterType type)
{
	Float3* d_inBuffer = nullptr;
	Float3* d_outBuffer = nullptr;

	Float3* h_inBuffer = new Float3[size.x * size.y];
	Float3* h_outBuffer = new Float3[size.x * size.y];

	GpuErrorCheck(cudaMalloc((void**)& d_inBuffer, size.x * size.y * sizeof(Float3)));
	GpuErrorCheck(cudaMalloc((void**)& d_outBuffer, size.x * size.y * sizeof(Float3)));

	ConvertUcharToFloat3(in, h_inBuffer, size);

	GpuErrorCheck(cudaMemcpy(d_inBuffer, h_inBuffer, size.x * size.y * sizeof(Float3), cudaMemcpyHostToDevice));

	switch (type)
	{
	case FilterType::Gaussian:
		CalculateGaussianKernel(2.0f);
		GaussianFilter <<< dim3(size.x / 8, size.y / 8, 1), dim3(8, 8, 1) >>> (d_inBuffer, d_outBuffer, size);
		break;
	case FilterType::Copy:
	default:
		Copy <<< dim3(size.x / 8, size.y / 8, 1), dim3(8, 8, 1) >>> (d_inBuffer, d_outBuffer, size);
	}

	GpuErrorCheck(cudaMemcpy(h_outBuffer, d_outBuffer, size.x * size.y * sizeof(Float3), cudaMemcpyDeviceToHost));

	ConvertFloat3ToUchar(out, h_outBuffer, size);

	GpuErrorCheck(cudaFree(d_inBuffer));
	GpuErrorCheck(cudaFree(d_outBuffer));

	delete h_inBuffer;
	delete h_outBuffer;
}

int main()
{
	int x, y, n;
	const char* filename = "lenna.png";
	uchar* in = stbi_load(filename, &x, &y, &n, 0);
	uchar* out = new uchar[x * y * n];

	RunFilter(in, out, Int2(x, y), FilterType::Copy);
	std::string filenameWrite = "result/copy.png";
	stbi_write_png(filenameWrite.c_str(), x, y, 3, (void*)out, 0);

	RunFilter(in, out, Int2(x, y), FilterType::Gaussian);
	filenameWrite = "result/gaussian.png";
	stbi_write_png(filenameWrite.c_str(), x, y, 3, (void*)out, 0);

	delete[] out;
	stbi_image_free(in);
	return 0;
}