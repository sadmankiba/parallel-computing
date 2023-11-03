#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include "count.cuh"
#include <cstdlib>

// Find the unique integers in the array d_in,
// store these integers in values array in ascending order,
// store the occurrences of these integers in counts array.
// values and counts should have equal length.
// Example:
// d_in = [3, 5, 1, 2, 3, 1]
// Expected output:
// values = [1, 2, 3, 5]
// counts = [2, 1, 2, 1]
void count(const thrust::device_vector<int>& d_in,
                 thrust::device_vector<int>& values,
                 thrust::device_vector<int>& counts){
    // sort
    thrust::device_vector<int> d(d_in.size());
    thrust::copy(d_in.begin(), d_in.end(), d.begin());
    thrust::sort(thrust::device, d.begin(), d.end());

    thrust::device_vector<int> freq(counts.size());
    thrust::fill(freq.begin(), freq.end(), 1);

    thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> nend;
    nend = thrust::reduce_by_key(thrust::device, d.begin(), d.end(), freq.begin(), values.begin(), counts.begin());
    int numVals = nend.first - values.begin();

    values.resize(numVals);
    counts.resize(numVals);
}