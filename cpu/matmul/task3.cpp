#include "matmul.h"
#include <chrono>
#include <iostream>

using namespace std;

int main() {
    const unsigned int n = 1024;
    double* A = (double *) malloc(n * n * sizeof(double));
    double* B = (double *) malloc(n * n * sizeof(double));
    double* C = (double *) malloc(n * n * sizeof(double));
    for (unsigned int i = 0; i < n * n; i++) {
        A[i] = (double) rand() / RAND_MAX * 2.0 - 1.0;
        B[i] = (double) rand() / RAND_MAX * 2.0 - 1.0;
    }
    cout << n << endl;

    std::chrono::high_resolution_clock::time_point start, stop;
    std::chrono::duration<double, std::milli> duration;
    start = chrono::high_resolution_clock::now();
    mmul1(A, B, C, n);
    stop = chrono::high_resolution_clock::now();

    duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << duration.count() << endl;
    cout << C[n * n - 1] << endl;

    start = chrono::high_resolution_clock::now();
    mmul2(A, B, C, n);
    stop = chrono::high_resolution_clock::now();

    duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << duration.count() << endl;
    cout << C[n * n - 1] << endl;

    start = chrono::high_resolution_clock::now();
    mmul3(A, B, C, n);
    stop = chrono::high_resolution_clock::now();

    duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << duration.count() << endl;
    cout << C[n * n - 1] << endl;

    //declare two vectors of size n*n
    vector<double> A_vec(n * n);
    vector<double> B_vec(n * n);
    for (unsigned int i = 0; i < n * n; i++) {
        A_vec[i] = A[i];
        B_vec[i] = B[i];
    }

    start = chrono::high_resolution_clock::now();
    mmul4(A_vec, B_vec, C, n);
    stop = chrono::high_resolution_clock::now();

    duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << duration.count() << endl;
    cout << C[n * n - 1] << endl;
    return 0;
}