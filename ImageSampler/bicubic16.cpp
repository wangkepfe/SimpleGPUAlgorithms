#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cmath>

class vec2f
{
public:
    vec2f() : x{0}, y{0} {}
    vec2f(float x, float y = 0.0f) : x{x}, y{y} {}
    vec2f operator + (float a) const { return vec2f(x + a, y + a); }
    vec2f operator - (float a) const { return vec2f(x - a, y - a); }
    vec2f operator * (float a) const { return vec2f(x * a, y * a); }
    vec2f operator / (float a) const { return vec2f(x / a, y / a); }
    vec2f operator + (const vec2f& v) const { return vec2f(x + v.x, y + v.y); }
    vec2f operator - (const vec2f& v) const { return vec2f(x - v.x, y - v.y); }
    vec2f operator * (const vec2f& v) const { return vec2f(x * v.x, y * v.y); }
    vec2f operator / (const vec2f& v) const { return vec2f(x / v.x, y / v.y); }
	vec2f operator += (const vec2f& v) { x += v.x; y += v.y; return *this; }
    vec2f operator -= (const vec2f& v) { x -= v.x; y -= v.y; return *this; }
    vec2f operator *= (const vec2f& v) { x *= v.x; y *= v.y; return *this; }
    vec2f operator /= (const vec2f& v) { x /= v.x; y /= v.y; return *this; }
    vec2f operator - () const { return vec2f(-x, -y); }
    float x, y;
};

class vec4f
{
public:
    vec4f() : x{0}, y{0}, z{0}, w{0} {}
    vec4f(float x, float y, float z, float w) : x{x}, y{y}, z{z}, w{w} {}
    vec4f(const vec2f& v1, const vec2f& v2) : x{v1.x}, y{v1.y}, z{v2.x}, w{v2.y} {}
    vec4f operator + (float a) const { return vec4f(x + a, y + a, z + a, w + a); }
    vec4f operator - (float a) const { return vec4f(x - a, y - a, z - a, w - a); }
    vec4f operator * (float a) const { return vec4f(x * a, y * a, z * a, w * a); }
    vec4f operator / (float a) const { return vec4f(x / a, y / a, z / a, w / a); }
    vec4f operator += (float a) { x += a; y += a; z += a; w += a; return *this; }
    vec4f operator -= (float a) { x -= a; y -= a; z -= a; w -= a; return *this; }
    vec4f operator *= (float a) { x *= a; y *= a; z *= a; w *= a; return *this; }
    vec4f operator /= (float a) { x /= a; y /= a; z /= a; w /= a; return *this; }
    vec4f operator + (const vec4f& v) const { return vec4f(x + v.x, y + v.y, z + v.z, w + v.w); }
    vec4f operator - (const vec4f& v) const { return vec4f(x - v.x, y - v.y, z - v.z, w - v.w); }
    vec4f operator * (const vec4f& v) const { return vec4f(x * v.x, y * v.y, z * v.z, w * v.w); }
    vec4f operator / (const vec4f& v) const { return vec4f(x / v.x, y / v.y, z / v.z, w / v.w); }
    vec4f operator += (const vec4f& v) { x += v.x; y += v.y; z += v.z; w += v.w; return *this; }
    vec4f operator -= (const vec4f& v) { x -= v.x; y -= v.y; z -= v.z; w -= v.w; return *this; }
    vec4f operator *= (const vec4f& v) { x *= v.x; y *= v.y; z *= v.z; w *= v.w; return *this; }
    vec4f operator /= (const vec4f& v) { x /= v.x; y /= v.y; z /= v.z; w /= v.w; return *this; }
    float x, y, z, w;
};

class vec2i
{
public:
    vec2i() : x{0}, y{0} {}
    vec2i(int x, int y) : x{x}, y{y} {}
    vec2i operator + (int a) const { return vec2i(x + a, y + a); }
    vec2i operator - (int a) const { return vec2i(x - a, y - a); }
    vec2i operator + (const vec2i& v) const { return vec2i(x + v.x, y + v.y); }
    vec2i operator - (const vec2i& v) const { return vec2i(x - v.x, y - v.y); }
    int x, y;
};

vec2i operator * (const vec2f& v, int a) { return vec2i(v.x * a, v.y * a); }
vec2i operator / (const vec2f& v, int a) { return vec2i(v.x / a, v.y / a); }
vec2f operator * (const vec2i& v, float a) { return vec2f(v.x * a, v.y * a); }
vec2f operator / (const vec2i& v, float a) { return vec2f(v.x / a, v.y / a); }
vec2f operator * (float a, const vec2f& v) { return vec2f(v.x * a, v.y * a); }
vec2f operator / (float a, const vec2f& v) { return vec2f(a / v.x, a / v.y); }
vec2f operator + (float a, const vec2f& v) { return vec2f(v.x + a, v.y + a); }
vec2f operator - (float a, const vec2f& v) { return vec2f(a - v.x, a - v.y); }
vec2f operator * (float a, const vec2i& v) { return vec2f((float)v.x * a, (float)v.y * a); }
vec2f operator / (float a, const vec2i& v) { return vec2f(a / (float)v.x, a / (float)v.y); }
vec2f operator * (const vec2f& vf, const vec2i& vi) { return vec2f(vf.x * vi.x, vf.y * vi.y); }
vec2f operator / (const vec2f& vf, const vec2i& vi) { return vec2f(vf.x / vi.x, vf.y / vi.y); }
vec2f operator * (const vec2i& vi, const vec2f& vf) { return vec2f(vi.x * vf.x, vi.y * vf.y); }
vec2f operator / (const vec2i& vi, const vec2f& vf) { return vec2f(vi.x / vf.x, vi.y / vf.y); }
vec4f operator * (float a, const vec4f& v) { return vec4f(v.x * a, v.y * a, v.z * a, v.w * a); }
vec4f operator / (float a, const vec4f& v) { return vec4f(a / v.x, a / v.y, a / v.z, a / v.w); }
vec2f floor(const vec2f& v) { return vec2f(std::floor(v.x), std::floor(v.y)); }
vec2i floori(const vec2f& v) { return vec2i(static_cast<int>(std::floor(v.x)), static_cast<int>(std::floor(v.y))); }
vec2f fract(const vec2f& v) { float intPart; return vec2f(std::modff(v.x, &intPart), std::modff(v.y, &intPart)); }
vec2i roundi(const vec2f& v) { return vec2i(static_cast<int>(std::round(v.x)), static_cast<int>(std::round(v.y))); }

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
}

void outputWrite(unsigned char* tex, const vec2i& pos, const vec2i& size, const vec4f& color)
{
    tex[(pos.x * size.x + pos.y) * 4 + 0] = postProcessFp32Uint8(color.x);
    tex[(pos.x * size.x + pos.y) * 4 + 1] = postProcessFp32Uint8(color.y);
    tex[(pos.x * size.x + pos.y) * 4 + 2] = postProcessFp32Uint8(color.z);
    tex[(pos.x * size.x + pos.y) * 4 + 3] = 255;
}

vec4f samplePoint(unsigned char* tex, const vec2i& pos, const vec2i& size)
{
    vec4f result;
    vec2i exterPos = pos;
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
    result = vec4f(static_cast<float>(tex[(exterPos.x * size.x + exterPos.y) * 4 + 0]) / 255.0f,
                   static_cast<float>(tex[(exterPos.x * size.x + exterPos.y) * 4 + 1]) / 255.0f,
                   static_cast<float>(tex[(exterPos.x * size.x + exterPos.y) * 4 + 2]) / 255.0f,
                   static_cast<float>(tex[(exterPos.x * size.x + exterPos.y) * 4 + 3]) / 255.0f);
	return result;
}

vec4f sampleNearest(unsigned char* in, const vec2f& uv, const vec2i& size)
{
    vec2f pos = uv * size;
    vec2i p00 = floori(pos - 0.5f);
    vec4f c00 = samplePoint(in, p00, size);
    return c00;
}

vec4f sampleBilinear(
    unsigned char* in,
    const vec2f& uv,
    const vec2i& size)
{
    vec2f UV = uv * size;
    vec2f invTexSize = 1.0f / size;
    vec2f tc = floor( UV - 0.5f ) + 0.5f;
    vec2f f = UV - tc;

    vec2f w0 = 1.0f - f;
    vec2f w1 = f;

    vec2i uv0 = floori(UV - 0.5f);
    vec2i uv1 = uv0 + 1;

    vec4f c00 = samplePoint(in, vec2i(uv0.x, uv0.y), size) * (w0.x * w0.y);
    vec4f c01 = samplePoint(in, vec2i(uv0.x, uv1.y), size) * (w0.x * w1.y);
    vec4f c10 = samplePoint(in, vec2i(uv1.x, uv0.y), size) * (w1.x * w0.y);
    vec4f c11 = samplePoint(in, vec2i(uv1.x, uv1.y), size) * (w1.x * w1.y);

    return c00 + c01 + c10 + c11;
}

vec4f sampleBicubic12(
    unsigned char*  tex,
    const    vec2f& uv,
    const    vec2i& texSize)
{
    vec2f Weight[3];
    vec2f Sample[3];
    vec2f SamplesUV[5];
    float SamplesWeight[5];

	vec2f UV = uv * texSize;
    vec2f InvSize = 1.0f / texSize;

	vec2f tc = floor( UV - 0.5f ) + 0.5f;
	vec2f f = UV - tc;
	vec2f f2 = f * f;
	vec2f f3 = f2 * f;

	vec2f w0 = f2 - 0.5f * (f3 + f);
	vec2f w1 = 1.5f * f3 - 2.5f * f2 + 1.0f;
	vec2f w3 = 0.5f * (f3 - f2);
	vec2f w2 = 1.0f - w0 - w1 - w3;

	Weight[0] = w0;
	Weight[1] = w1 + w2;
	Weight[2] = w3;

	Sample[0] = tc - 1;
	Sample[1] = tc + w2 / Weight[1];
	Sample[2] = tc + 2;

	Sample[0] *= InvSize;
	Sample[1] *= InvSize;
	Sample[2] *= InvSize;

    SamplesUV[0] = vec2f(Sample[1].x, Sample[0].y);
	SamplesUV[1] = vec2f(Sample[0].x, Sample[1].y);
	SamplesUV[2] = vec2f(Sample[1].x, Sample[1].y);
	SamplesUV[3] = vec2f(Sample[2].x, Sample[1].y);
	SamplesUV[4] = vec2f(Sample[1].x, Sample[2].y);

    SamplesWeight[0] = Weight[1].x * Weight[0].y;
	SamplesWeight[1] = Weight[0].x * Weight[1].y;
	SamplesWeight[2] = Weight[1].x * Weight[1].y;
	SamplesWeight[3] = Weight[2].x * Weight[1].y;
	SamplesWeight[4] = Weight[1].x * Weight[2].y;

	vec4f OutColor;
    float sumWeight;
	for (int i = 0; i < 5; i++)
	{
        sumWeight += SamplesWeight[i];
		OutColor += sampleBilinear(tex, SamplesUV[i], texSize) * SamplesWeight[i];
	}
	OutColor /= sumWeight;

	return OutColor;
}

vec4f sampleBicubic16(
    unsigned char*  tex,
    const    vec2f& uv,
    const    vec2i& texSize)
{
    vec2f UV = uv * texSize;
    vec2f invTexSize = 1.0f / texSize;
    vec2f tc = floor( UV - 0.5f ) + 0.5f;
    vec2f f = UV - tc;
    vec2f f2 = f * f;
	vec2f f3 = f2 * f;

	vec2f w0 = f2 - 0.5f * (f3 + f);
	vec2f w1 = 1.5f * f3 - 2.5f * f2 + 1.0f;
	vec2f w3 = 0.5f * (f3 - f2);
	vec2f w2 = 1.0f - w0 - w1 - w3;

    vec2i tc1 = floori(UV - 0.5f);
    vec2i tc0 = tc1 - 1;
    vec2i tc2 = tc1 + 1;
    vec2i tc3 = tc1 + 2;

    vec2i sampleUV[16] = {
        { tc0.x, tc0.y },
        { tc1.x, tc0.y },
        { tc2.x, tc0.y },
        { tc3.x, tc0.y },

        { tc0.x, tc1.y },
        { tc1.x, tc1.y },
        { tc2.x, tc1.y },
        { tc3.x, tc1.y },

        { tc0.x, tc2.y },
        { tc1.x, tc2.y },
        { tc2.x, tc2.y },
        { tc3.x, tc2.y },

        { tc0.x, tc3.y },
        { tc1.x, tc3.y },
        { tc2.x, tc3.y },
        { tc3.x, tc3.y },
    };

    float weights[16] = {
        w0.x * w0.y,
        w1.x * w0.y,
        w2.x * w0.y,
        w3.x * w0.y,

        w0.x * w1.y,
        w1.x * w1.y,
        w2.x * w1.y,
        w3.x * w1.y,

        w0.x * w2.y,
        w1.x * w2.y,
        w2.x * w2.y,
        w3.x * w2.y,

        w0.x * w3.y,
        w1.x * w3.y,
        w2.x * w3.y,
        w3.x * w3.y,
    };

	vec4f OutColor;
    float sumWeight = 0;
	for (int i = 0; i < 16; i++)
	{
        sumWeight += weights[i];
		OutColor += samplePoint(tex, sampleUV[i], texSize) * weights[i];
	}
	OutColor /= sumWeight;

    return OutColor;
}

int main()
{
    const vec2i texSize(5, 5);
    const vec2i outSize(100, 100);

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
            vec2i outUV(i, j);
            vec2f uv = vec2i(i, j) / static_cast<float>(outSize.x);

            vec4f color = sampleBicubic16(in, uv, texSize);

            outputWrite(out, outUV, outSize, color);
        }
    }
    const char* filenameWrite = "output/bicubic16.png";
    stbi_write_png(filenameWrite, outSize.x, outSize.y, 4, (void*)out, 0);
    delete[] out;
    stbi_image_free(in);
    return 0;
}