#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <string>
#include <cmath>
#define PI 3.14159265358979323846

class float2
{
public:
    float2() : x{0}, y{0} {}
    float2(float x) : x{x}, y{x} {}
    float2(float x, float y) : x{x}, y{y} {}
    float2 operator + (float a) const { return float2(x + a, y + a); }
    float2 operator - (float a) const { return float2(x - a, y - a); }
    float2 operator * (float a) const { return float2(x * a, y * a); }
    float2 operator / (float a) const { return float2(x / a, y / a); }
    float2 operator + (const float2& v) const { return float2(x + v.x, y + v.y); }
    float2 operator - (const float2& v) const { return float2(x - v.x, y - v.y); }
    float2 operator * (const float2& v) const { return float2(x * v.x, y * v.y); }
    float2 operator / (const float2& v) const { return float2(x / v.x, y / v.y); }
	float2 operator += (const float2& v) { x += v.x; y += v.y; return *this; }
    float2 operator -= (const float2& v) { x -= v.x; y -= v.y; return *this; }
    float2 operator *= (const float2& v) { x *= v.x; y *= v.y; return *this; }
    float2 operator /= (const float2& v) { x /= v.x; y /= v.y; return *this; }
    float2 operator - () const { return float2(-x, -y); }
    float x, y;
};

class float4
{
public:
    float4() : x{0}, y{0}, z{0}, w{0} {}
    float4(float x, float y, float z, float w) : x{x}, y{y}, z{z}, w{w} {}
    float4(const float2& v1, const float2& v2) : x{v1.x}, y{v1.y}, z{v2.x}, w{v2.y} {}
    float4 operator + (float a) const { return float4(x + a, y + a, z + a, w + a); }
    float4 operator - (float a) const { return float4(x - a, y - a, z - a, w - a); }
    float4 operator * (float a) const { return float4(x * a, y * a, z * a, w * a); }
    float4 operator / (float a) const { return float4(x / a, y / a, z / a, w / a); }
    float4 operator += (float a) { x += a; y += a; z += a; w += a; return *this; }
    float4 operator -= (float a) { x -= a; y -= a; z -= a; w -= a; return *this; }
    float4 operator *= (float a) { x *= a; y *= a; z *= a; w *= a; return *this; }
    float4 operator /= (float a) { x /= a; y /= a; z /= a; w /= a; return *this; }
    float4 operator + (const float4& v) const { return float4(x + v.x, y + v.y, z + v.z, w + v.w); }
    float4 operator - (const float4& v) const { return float4(x - v.x, y - v.y, z - v.z, w - v.w); }
    float4 operator * (const float4& v) const { return float4(x * v.x, y * v.y, z * v.z, w * v.w); }
    float4 operator / (const float4& v) const { return float4(x / v.x, y / v.y, z / v.z, w / v.w); }
    float4 operator += (const float4& v) { x += v.x; y += v.y; z += v.z; w += v.w; return *this; }
    float4 operator -= (const float4& v) { x -= v.x; y -= v.y; z -= v.z; w -= v.w; return *this; }
    float4 operator *= (const float4& v) { x *= v.x; y *= v.y; z *= v.z; w *= v.w; return *this; }
    float4 operator /= (const float4& v) { x /= v.x; y /= v.y; z /= v.z; w /= v.w; return *this; }
    float operator [] (int i) const { return v[i]; }
    float& operator [] (int i) { return v[i]; }
    union {
        struct { float x, y, z, w; };
        float v[4];
    };
};

class int2
{
public:
    int2() : x{0}, y{0} {}
    int2(int x, int y) : x{x}, y{y} {}
    int2 operator + (int a) const { return int2(x + a, y + a); }
    int2 operator - (int a) const { return int2(x - a, y - a); }
    int2 operator + (const int2& v) const { return int2(x + v.x, y + v.y); }
    int2 operator - (const int2& v) const { return int2(x - v.x, y - v.y); }
    int x, y;
};

int2 operator * (const float2& v, int a) { return int2(v.x * a, v.y * a); }
int2 operator / (const float2& v, int a) { return int2(v.x / a, v.y / a); }
float2 operator * (const int2& v, float a) { return float2(v.x * a, v.y * a); }
float2 operator / (const int2& v, float a) { return float2(v.x / a, v.y / a); }
float2 operator * (float a, const float2& v) { return float2(v.x * a, v.y * a); }
float2 operator / (float a, const float2& v) { return float2(a / v.x, a / v.y); }
float2 operator + (float a, const float2& v) { return float2(v.x + a, v.y + a); }
float2 operator - (float a, const float2& v) { return float2(a - v.x, a - v.y); }
float2 operator * (float a, const int2& v) { return float2((float)v.x * a, (float)v.y * a); }
float2 operator / (float a, const int2& v) { return float2(a / (float)v.x, a / (float)v.y); }
float2 operator * (const float2& vf, const int2& vi) { return float2(vf.x * vi.x, vf.y * vi.y); }
float2 operator / (const float2& vf, const int2& vi) { return float2(vf.x / vi.x, vf.y / vi.y); }
float2 operator * (const int2& vi, const float2& vf) { return float2(vi.x * vf.x, vi.y * vf.y); }
float2 operator / (const int2& vi, const float2& vf) { return float2(vi.x / vf.x, vi.y / vf.y); }
float4 operator * (float a, const float4& v) { return float4(v.x * a, v.y * a, v.z * a, v.w * a); }
float4 operator / (float a, const float4& v) { return float4(a / v.x, a / v.y, a / v.z, a / v.w); }
float2 operator * (const int2& v, const int2& v2) { return float2(v.x * (float)v2.x, v.y * (float)v2.y); }
float2 operator / (const int2& v, const int2& v2) { return float2(v.x / (float)v2.x, v.y / (float)v2.y); }
float2 floor(const float2& v) { return float2(std::floor(v.x), std::floor(v.y)); }
int2 floori(const float2& v) { return int2(static_cast<int>(std::floor(v.x)), static_cast<int>(std::floor(v.y))); }
float2 fract(const float2& v) { float intPart; return float2(std::modf(v.x, &intPart), std::modf(v.y, &intPart)); }
int2 roundi(const float2& v) { return int2(static_cast<int>(std::round(v.x)), static_cast<int>(std::round(v.y))); }
float2 abs(const float2& v) { return float2(std::abs(v.x), std::abs(v.y)); }
float2 sin(const float2& v) { return float2(std::sin(v.x), std::sin(v.y)); }

enum OutOfBorderMode {
    CLAMP_TO_BORDER,
    REPEAT,
};

template<class T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi)
{
	assert(!(hi < lo));
	return (v < lo) ? lo : (hi < v) ? hi : v;
}

unsigned char postProcessFp32Uint8(float inColor) {
    const float exposure = 1.0f;
    const float gamma = 2.2f;
    // no negative color
    float mapped = inColor < 0 ? 0 : inColor;
    // exposure tone mapping
    mapped = 1.0f - std::exp(-mapped * exposure);
    // Gamma correction
	mapped = std::pow(mapped, 1.0f / gamma);
    // fp32 to uint8
    return static_cast<unsigned char>(clamp(mapped, 0.0f, 1.0f) * 255.0f + 0.5f);
    //return static_cast<unsigned char>(clamp(inColor, 0.0f, 1.0f) * 255.0f + 0.5f);
}

float preprocessUint8Fp32(unsigned char inColor) {
	const float gamma = 2.2f;
	float outColor = static_cast<float>(inColor);
	outColor /= 255.0f;
	outColor = std::pow(outColor, gamma);
	return outColor;
}

void outputWrite(unsigned char* tex, const int2& pos, const int2& size, const float4& color)
{
    tex[(pos.x * size.x + pos.y) * 4 + 0] = postProcessFp32Uint8(color.x);
    tex[(pos.x * size.x + pos.y) * 4 + 1] = postProcessFp32Uint8(color.y);
    tex[(pos.x * size.x + pos.y) * 4 + 2] = postProcessFp32Uint8(color.z);
    tex[(pos.x * size.x + pos.y) * 4 + 3] = 255;
}

float4 samplePoint(unsigned char* tex, const int2& pos, const int2& size)
{
    float4 result;
    int2 exterPos = pos;

    OutOfBorderMode mode = CLAMP_TO_BORDER;

    if (mode == CLAMP_TO_BORDER) {
        if (exterPos.y < 0)       { exterPos.y = 0; }
        if (exterPos.x < 0)       { exterPos.x = 0; }
        if (exterPos.x >= size.x) { exterPos.x = size.x - 1; }
        if (exterPos.y >= size.y) { exterPos.y = size.y - 1; }
    }
    else if (mode == REPEAT) {
        if (exterPos.x < 0)       { exterPos.x = size.x - (-exterPos.x) % size.x; }
        if (exterPos.y < 0)       { exterPos.y = size.y - (-exterPos.y) % size.y; }
        if (exterPos.x >= size.x) { exterPos.x = exterPos.x % size.x; }
        if (exterPos.y >= size.y) { exterPos.y = exterPos.y % size.y; }
    }

    result = float4(
		preprocessUint8Fp32(tex[(exterPos.x * size.x + exterPos.y) * 4 + 0]),
		preprocessUint8Fp32(tex[(exterPos.x * size.x + exterPos.y) * 4 + 1]),
		preprocessUint8Fp32(tex[(exterPos.x * size.x + exterPos.y) * 4 + 2]),
		preprocessUint8Fp32(tex[(exterPos.x * size.x + exterPos.y) * 4 + 3]));

	return result;
}

float4 sampleNearest(unsigned char* tex, const float2& uv, const int2& size)
{
    // float2 pos = uv * (size - 1);
    // int2 p00 = roundi(pos);
    float2 pos = uv * size;
    int2 p = floori(pos);
    float4 c = samplePoint(tex, p, size);
    return c;
}

float4 sampleBilinear(
    unsigned char* in,
    const float2& uv,
    const int2& size)
{
    float2 UV = uv * size;
    float2 tc = floor( UV - 0.5f ) + 0.5f;
    float2 f = UV - tc; // [0,1)

    float2 w0 = 1.0f - f;
    float2 w1 = f;

    int2 uv0 = floori(UV - 0.5f);
    int2 uv1 = uv0 + 1;

    float4 c00 = samplePoint(in, int2(uv0.x, uv0.y), size) * (w0.x * w0.y);
    float4 c01 = samplePoint(in, int2(uv0.x, uv1.y), size) * (w0.x * w1.y);
    float4 c10 = samplePoint(in, int2(uv1.x, uv0.y), size) * (w1.x * w0.y);
    float4 c11 = samplePoint(in, int2(uv1.x, uv1.y), size) * (w1.x * w1.y);

    return c00 + c01 + c10 + c11;
}

float4 sampleBicubic12(
    unsigned char*  tex,
    const    float2& uv,
    const    int2& texSize)
{
    float2 Weight[3];
    float2 Sample[3];
    float2 SamplesUV[5];
    float SamplesWeight[5];

	float2 UV = uv * texSize;
    float2 InvSize = 1.0f / texSize;

	float2 tc = floor( UV - 0.5f ) + 0.5f;
	float2 f = UV - tc;
	float2 f2 = f * f;
	float2 f3 = f2 * f;

	float2 w0 = f2 - 0.5f * (f3 + f);
	float2 w1 = 1.5f * f3 - 2.5f * f2 + 1.0f;
	float2 w3 = 0.5f * (f3 - f2);
	float2 w2 = 1.0f - w0 - w1 - w3;

	Weight[0] = w0;
	Weight[1] = w1 + w2;
	Weight[2] = w3;

	Sample[0] = tc - 1;
	Sample[1] = tc + w2 / Weight[1];
	Sample[2] = tc + 2;

	Sample[0] *= InvSize;
	Sample[1] *= InvSize;
	Sample[2] *= InvSize;

    SamplesUV[0] = float2(Sample[1].x, Sample[0].y);
	SamplesUV[1] = float2(Sample[0].x, Sample[1].y);
	SamplesUV[2] = float2(Sample[1].x, Sample[1].y);
	SamplesUV[3] = float2(Sample[2].x, Sample[1].y);
	SamplesUV[4] = float2(Sample[1].x, Sample[2].y);

    SamplesWeight[0] = Weight[1].x * Weight[0].y;
	SamplesWeight[1] = Weight[0].x * Weight[1].y;
	SamplesWeight[2] = Weight[1].x * Weight[1].y;
	SamplesWeight[3] = Weight[2].x * Weight[1].y;
	SamplesWeight[4] = Weight[1].x * Weight[2].y;

	float4 OutColor;
    float sumWeight = 0;
	for (int i = 0; i < 5; i++)
	{
        sumWeight += SamplesWeight[i];
		OutColor += sampleBilinear(tex, SamplesUV[i], texSize) * SamplesWeight[i];
	}
	OutColor /= sumWeight;

	return OutColor;
}

float4 sampleBicubic16(
    unsigned char*  tex,
    const    float2& uv,
    const    int2& texSize)
{
    float2 UV = uv * texSize;
    float2 invTexSize = 1.0f / texSize;
    float2 tc = floor( UV - 0.5f ) + 0.5f;
    float2 f = UV - tc;
    float2 f2 = f * f;
	float2 f3 = f2 * f;

	float2 w0 = f2 - 0.5f * (f3 + f);
	float2 w1 = 1.5f * f3 - 2.5f * f2 + 1.0f;
	float2 w3 = 0.5f * (f3 - f2);
	float2 w2 = 1.0f - w0 - w1 - w3;

    int2 tc1 = floori(UV - 0.5f);
    int2 tc0 = tc1 - 1;
    int2 tc2 = tc1 + 1;
    int2 tc3 = tc1 + 2;

    int2 sampleUV[16] = {
        { tc0.x, tc0.y }, { tc1.x, tc0.y }, { tc2.x, tc0.y }, { tc3.x, tc0.y },
        { tc0.x, tc1.y }, { tc1.x, tc1.y }, { tc2.x, tc1.y }, { tc3.x, tc1.y },
        { tc0.x, tc2.y }, { tc1.x, tc2.y }, { tc2.x, tc2.y }, { tc3.x, tc2.y },
        { tc0.x, tc3.y }, { tc1.x, tc3.y }, { tc2.x, tc3.y }, { tc3.x, tc3.y },
    };

    float weights[16] = {
        w0.x * w0.y,  w1.x * w0.y,  w2.x * w0.y,  w3.x * w0.y,
        w0.x * w1.y,  w1.x * w1.y,  w2.x * w1.y,  w3.x * w1.y,
        w0.x * w2.y,  w1.x * w2.y,  w2.x * w2.y,  w3.x * w2.y,
        w0.x * w3.y,  w1.x * w3.y,  w2.x * w3.y,  w3.x * w3.y,
    };

	float4 OutColor;
    float sumWeight = 0;
	for (int i = 0; i < 16; i++)
	{
        sumWeight += weights[i];
		OutColor += samplePoint(tex, sampleUV[i], texSize) * weights[i];
	}
	OutColor /= sumWeight;

    return OutColor;
}

float2 smoothStep(const float2& f)
{
    float2 f2 = f * f;
	float2 f3 = f2 * f;
    return -2.0f * f3 + 3.0f * f2;
}

float2 smootherStep(const float2& f)
{
    float2 f2 = f * f;
	float2 f3 = f2 * f;
    float2 f4 = f2 * f2;
    float2 f5 = f2 * f3;
    return 6.0f * f5 - 15.0f * f4 + 10.0f * f3;
}

float2 sigmoidAbs(const float2& f, float k)
{
    float2 x = f;
    // scale
    x = 2.0f * x - 1.0f;
    // func
    float2 nom   = (1.0f - k) * x;
    float2 denom = (-2.0f * k) * abs(x) + (k + 1.0f);
    float2 fx = nom / denom;
    // scale
    fx = (fx + 1.0f) / 2.0f;
    return fx;
}

float2 sinusoidalStep(const float2& f)
{
	return (sin(PI * (f - 0.5f)) + 1.0f) * 0.5f;
}

float4 sampleBicubic4(
    unsigned char*  tex,
    const    float2& uv,
    const    int2& texSize)
{
    float2 UV = uv * texSize;
    float2 invTexSize = 1.0f / texSize;
    float2 tc = floor( UV - 0.5f ) + 0.5f;
    float2 f = UV - tc;

    float2 w1 = smoothStep(f);
    //float2 w1 = smootherStep(f);
    //float2 w1 = sigmoidAbs(f, -0.5f);
    //float2 w1 = sinusoidalStep(f);
    float2 w0 = 1.0f - w1;

    int2 tc0 = floori(UV - 0.5f);
    int2 tc1 = tc0 + 1;

    int2 sampleUV[4] = {
        { tc0.x, tc0.y }, { tc1.x, tc0.y },
        { tc0.x, tc1.y }, { tc1.x, tc1.y },
    };

    float weights[4] = {
        w0.x * w0.y,  w1.x * w0.y,
        w0.x * w1.y,  w1.x * w1.y,
    };

	float4 OutColor;
    float sumWeight = 0;
	for (int i = 0; i < 4; i++)
	{
        sumWeight += weights[i];
		OutColor += samplePoint(tex, sampleUV[i], texSize) * weights[i];
	}
	OutColor /= sumWeight;

    return OutColor;
}

float4 sampleAlterSignedBilinear(
    unsigned char* in,
    const float2& uv,
    const int2& size)
{
    float2 UV = uv * size;
    float2 tc = floor( UV - 0.5f ) + 0.5f;
    float2 f = UV - tc; // [0,1)

    float2 w0 = f - 1.0f;
    float2 w1 = f;

    int2 uv0 = floori(UV - 0.5f);
    int2 uv1 = uv0 + 1;

    float4 c00 = samplePoint(in, int2(uv0.x, uv0.y), size) * (w0.x * w0.y);
    float4 c01 = samplePoint(in, int2(uv0.x, uv1.y), size) * (w0.x * w1.y);
    float4 c10 = samplePoint(in, int2(uv1.x, uv0.y), size) * (w1.x * w0.y);
    float4 c11 = samplePoint(in, int2(uv1.x, uv1.y), size) * (w1.x * w1.y);

    float4 outColor = c00 + c01 + c10 + c11;

    return outColor;
}

float4 sampleBicubic4Linear(
    unsigned char*  tex,
    const    float2& uv,
    const    int2& texSize)
{
    float2 UV = uv * texSize;
    float2 tc = floor( UV - 0.5f ) + 0.49999f;
    float2 f = UV - tc;

    float2 s1  = ( 0.5f * f - 0.5f) * f;             // = w1 / (1 - f)
    float2 s12 = (-2.0f * f + 1.5f) * f + 1.0f;      // = (w2 - w1) / (1 - f)
    float2 s34 = ( 2.0f * f - 2.5f) * f - 0.5f;      // = (w4 - w3) / f

    float2 tc0 = (-f * s12 + s1      ) / (s12 * texSize) + uv; // tc - 1.0 + w2 / (w2 - w1)
    float2 tc1 = (-f * s34 + s1 + s34) / (s34 * texSize) + uv; // tc + 1.0 + w4 / (w4 - w3)

    float2 w0 = -f * s12 + s12; // w2 - w1
    float2 w1 = s34 * f;        // w4 - w3

    float2 sampleUV[4] = {
        { tc0.x, tc0.y }, { tc1.x, tc0.y },
        { tc0.x, tc1.y }, { tc1.x, tc1.y },
    };

    float4 weights {
        w0.x * w0.y,  w1.x * w0.y,
		w0.x * w1.y,  w1.x * w1.y
    };

    float4 OutColor;
	for (int i = 0; i < 4; i++)
	{
		OutColor += sampleAlterSignedBilinear(tex, sampleUV[i], texSize) * weights[i];
	}

	return OutColor;
}

int main()
{
    const int2 texSize(5, 5);
    const int2 outSize(1024, 1024);

    int x, y, n;
    const char* filename = "resources/input.png";
    unsigned char* in = stbi_load(filename, &x, &y, &n, 0);
	size_t outTotalSize = size_t(outSize.x) * size_t(outSize.y) * 4;
    unsigned char* out = new unsigned char[outTotalSize];
    memset(out, 0, outTotalSize);

    for (int i = 0; i < outSize.x; ++i)
	{
		for (int j = 0; j < outSize.y; ++j)
		{
			int2 outUV(i, j);
			float2 uv = int2(i, j) / outSize;
            //float4 color = sampleNearest(in, uv, texSize);
            //float4 color = sampleBilinear(in, uv, texSize);
            //float4 color = sampleBicubic16(in, uv, texSize);
            //float4 color = sampleBicubic12(in, uv, texSize);
            //float4 color = sampleBicubic4(in, uv, texSize);
            float4 color = sampleBicubic4Linear(in, uv, texSize);
			outputWrite(out, outUV, outSize, color);
		}
	}
    std::string filenameWrite = "output/bicubic4Linear.png";
	stbi_write_png(filenameWrite.c_str(), outSize.x, outSize.y, 4, (void*)out, 0);

    delete[] out;
    stbi_image_free(in);

    return 0;
}