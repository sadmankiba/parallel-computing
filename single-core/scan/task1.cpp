#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "scan.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: ./task1 <n>" << endl;
        return 1;
    }
    int p = stoi(argv[1]);
    // n = 2^p
    int n = 1 << p;

    float* array = (float*) malloc(n * sizeof(float));
    float* output = (float*) malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        array[i] = (float) rand() / RAND_MAX * 2.0 - 1.0;
    }
    chrono::duration<double, std::milli> duration;

    // Calling once to warm up the cache
    scan(array, output, n);

    auto start = chrono::high_resolution_clock::now();
    scan(array, output, n);
    auto stop = chrono::high_resolution_clock::now();

    // auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    duration = chrono::duration<double, std::milli>(stop - start);

    cout << duration.count() << endl;
    cout << array[0] << endl;
    cout << array[n - 1] << endl;

    free(array);
    free(output);

    return 0;
}
