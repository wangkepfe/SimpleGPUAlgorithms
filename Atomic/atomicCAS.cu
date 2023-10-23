
__device__ double atomicAdd(double *address, double val)
{
    unsigned long long int *address_as_ull =
        (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ void atomicMinAndConditionalWrite(float *dest1, float *dest2, float val1, float val2)
{
    unsigned int *destAddr1 = (unsigned int *)dest1;
    unsigned int oldVal1 = *destAddr1;
    bool minValUpdated = true;

    // Update the minimium value of dest1 to min(dest1, val1)
    do
    {
        unsigned int assumedVal1 = oldVal1;
        if (__uint_as_float(assumedVal1) > val1)
        {
            unsigned int newVal1 = __float_as_uint(val1);
            oldVal1 = atomicCAS(destAddr1, assumedVal1, newVal1);
        }
        else
        {
            minValUpdated = false;
            break;
        }
    } while (assumedVal1 != oldVal1);

    // If the minimun value is updated, then assign val2 to dest2
    if (minValUpdated)
    {
        unsigned int *destAddr2 = (unsigned int *)dest2;
        unsigned int oldVal2 = *destAddr2;
        do
        {
            unsigned int assumedVal2 = oldVal2;
            unsigned int newVal2 = __float_as_uint(val2);
            oldVal2 = atomicCAS(destAddr2, assumedVal2, newVal2);
        } while (assumedVal2 != oldVal2);
    }
}