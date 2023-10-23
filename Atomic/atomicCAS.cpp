#include <vector>
#include <atomic>
#include <iostream>
#include <thread>
#include <random>

using namespace std;

constexpr int testSize = 50;
constexpr int loopNum = 10;

atomic<int> threadCounter = 0;

float maxValue = 0;
int idxMax = 0;

float groundTruth;
int correctCaseNum = 0;

float sumValue = 0;
atomic<float> atomicSumValue = 0;

vector<float> inputBuffer;

atomic<float> atomicMax = 0;
atomic<int> atomicIdxMax = 0;

void atomicAdd(atomic<float> &dest, float val)
{
    bool succeeded = false;
    float oldVal = dest.load();

    while (!succeeded)
    {
        float newVal = oldVal + val;
        succeeded = dest.compare_exchange_weak(oldVal, newVal);
    }
}

void atomicMaxAndConditionalWrite(atomic<float> &dest1, atomic<int> &dest2, float val1, int val2)
{
    bool succeeded = false;
    float oldVal1 = dest1.load();
    while (!succeeded)
    {
        if (val1 > oldVal1)
        {
            float newVal1 = val1;
            succeeded = dest1.compare_exchange_weak(oldVal1, newVal1);
        }
        else
        {
            break;
        }
    }
    if (succeeded)
    {
        bool succeeded2 = false;
        int oldVal2 = dest2.load();
        while (!succeeded2)
        {
            int newVal2 = val2;
            succeeded2 = dest2.compare_exchange_weak(oldVal2, newVal2);
        }
    }
}

void compareAndWrite(float val, int threadId)
{
    if (val > maxValue)
    {
        maxValue = val;
        idxMax = threadId;
    }

    atomicMaxAndConditionalWrite(atomicMax, atomicIdxMax, val, threadId);
}

void sum(float val)
{
    sumValue += val;
    // atomicSumValue.fetch_add(val, std::memory_order_relaxed);
    atomicAdd(atomicSumValue, val);
}

void kernel(int threadId)
{
    float val = inputBuffer[threadId];
    compareAndWrite(val, threadId);
    sum(val);
}

void concurrentKernelWrapper(int threadId)
{
    threadCounter++;
    while (threadCounter.load() != testSize)
    {
    }

    kernel(threadId);
}

void singleThread()
{
    maxValue = 0;
    idxMax = 0;
    sumValue = 0;

    for (int i = 0; i < testSize; ++i)
    {
        kernel(i);
    }

    cout << "singleThread " << idxMax << " " << maxValue << " " << sumValue << "\n";
}

void multiThread()
{
    maxValue = 0;
    idxMax = 0;
    sumValue = 0;
    threadCounter.store(0);
    atomicSumValue.store(0);
    atomicMax.store(0);
    atomicIdxMax.store(0);

    vector<thread> threads;
    for (int i = 0; i < testSize; ++i)
    {
        threads.emplace_back(concurrentKernelWrapper, i);
    }

    for (auto &th : threads)
    {
        th.join();
    }

    cout << "singleThread " << idxMax << " " << maxValue << " " << sumValue << " " << atomicSumValue.load() << " " << atomicMax.load() << " " << atomicIdxMax.load() << "\n";

    if (groundTruth == maxValue)
    {
        correctCaseNum++;
    }
}

void main()
{
    default_random_engine generator;
    generator.seed(123);

    normal_distribution<float> distribution(10.0, 5.0);
    inputBuffer.resize(testSize);
    for (int i = 0; i < testSize; ++i)
    {
        float val = distribution(generator);
        inputBuffer[i] = val;
    }

    singleThread();

    groundTruth = maxValue;

    for (int i = 0; i < loopNum; ++i)
    {
        multiThread();
    }

    cout << "correctCaseNum " << correctCaseNum << "\n";
}