#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "msort.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: ./task3 n t ts" << endl;
        return 1;
    }
    int N = stoi(argv[1]);
    int n_threads = stoi(argv[2]);
    int thres = stoi(argv[3]);

    int* arr = (int*) malloc(N * sizeof(int));
    
    for (int i = 0; i < N; i++) {
        arr[i] = (int) (rand() % 2000) - 1000;
    }
    
    // for (int i = 0; i < N; i++) {
    //     cout << arr[i] << " ";
    // }
    // cout << endl;
    chrono::duration<double, std::milli> duration;

    auto start = chrono::high_resolution_clock::now();
    #pragma omp parallel num_threads(n_threads)
    {
        #pragma omp single
        msort(arr, N, thres);
    }
    auto stop = chrono::high_resolution_clock::now();

    duration = chrono::duration<double, std::milli>(stop - start);
    // for (int i = 0; i < N; i++) {
    //     cout << arr[i] << " ";
    // }
    // cout << endl;
    
    cout << arr[0] << endl;
    cout << arr[N - 1] << endl;
    cout << duration.count() << endl;

    free(arr);
    return 0;
}
