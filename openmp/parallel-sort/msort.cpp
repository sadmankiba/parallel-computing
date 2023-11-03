#include <iostream>
#include <mutex>
#include <chrono>

/*
single thread top-down mergesort
mergeSort(A, low, high, B)
    mid = (low + high) / 2
    mergeSort(A, low, mid, B)
    mergeSort(A, mid + 1, high, B)
    merge(A, low, mid, high, B)

merge(A, low, mid, high, B)
    i = low
    j = mid + 1
    for (int k = low; k <= high; k++)
        if (i > mid)
            B[k] = A[j++]
        else if (j > high)
            B[k] = A[i++]
        else if (A[i] < A[j])
            B[k] = A[i++]
        else
            B[k] = A[j++]
    for (int k = low; k <= high; k++)
        A[k] = B[k]
*/


/*
single thread bottom-up mergesort
treat each element as a sorted list of length 1 and merge them

mergeSort(A, low, high, B)
    for (int width = 1; width < n; width *= 2)
        for (int i = 0; i < n; i += 2 * width)
            merge(A, i, min(i + width, n), min(i + 2 * width, n), B)
*/

void msort_single(int *arr, size_t low, size_t high, int *B) {
    if (low >= high) return;
    size_t mid = (low + high) / 2;
    msort_single(arr, low, mid, B);
    msort_single(arr, mid + 1, high, B);
    
    size_t i = low, j = mid + 1;
    for (size_t k = low; k <= high; k++) {
        if (i > mid) {
            B[k] = arr[j++];
        } else if (j > high) {
            B[k] = arr[i++];
        } else if (arr[i] < arr[j]) {
            B[k] = arr[i++];
        } else {
            B[k] = arr[j++];
        }
    }
    for (size_t k = low; k <= high; k++) {
        arr[k] = B[k];
    }
}


std::mutex print_mutex;

/*
multi-thread top-down serial merge mergesort
mergeSort(A, low, high, B)
    mid = (low + high) / 2
    spawn mergeSort(A, low, mid, B)
    spawn mergeSort(A, mid + 1, high, B)
    sync
    merge(A, low, mid, high, B)
*/

void insertion_sort_inp(int* arr, int low, int high);
void insertion_sort(const int* arr, int low, int high, int *B);

void msort_multi(int *arr, size_t low, size_t high, int *B, size_t threshold) {
    if (low >= high) return;
    if ((high - low + 1) < threshold) {
        insertion_sort_inp(arr, low, high);
        // for (size_t k = low; k <= high; k++) {
        //     std::cout << arr[k] << " ";
        // }
        // std::cout << std::endl;
        return;
    }
    
    size_t mid = (low + high) / 2;
    #pragma omp task
    {
        msort_multi(arr, low, mid, B, threshold);
    }
    #pragma omp task
    {
        msort_multi(arr, mid + 1, high, B, threshold);
    }
    #pragma omp taskwait
    size_t i = low, j = mid + 1;
    for (size_t k = low; k <= high; k++) {
        if (i > mid) {
            B[k] = arr[j++];
        } else if (j > high) {
            B[k] = arr[i++];
        } else if (arr[i] < arr[j]) {
            B[k] = arr[i++];
        } else {
            B[k] = arr[j++];
        }
    }
    for (size_t k = low; k <= high; k++) {
        arr[k] = B[k];
    }
}

/* 
multi-thread top-down parallel merge mergesort
The idea of p-merge is split - AL: <= x, x, >= x, AR: <x, >=x. Merge AL <=x with AR <x and AL >=x with AR >=x. 

mergeSort(A, low, high, B)
    mid = (low + high) / 2
    spawn mergeSort(A, low, mid, B)
    spawn mergeSort(A, mid + 1, high, B)
    sync
    p-merge(A, low, mid, mid+1, high, B)

p-merge(A, lowL, highL, lowR, highR, B)
    midL = (lowL + highL) / 2
    posR = binarySearch(A[midL], lowR, highR)
    spawn p-merge(A, lowL, midL, lowR, posR - 1, B)
    spawn p-merge(A, midL + 1, highL, posR, highR, B)
    sync
*/
void mergep(int *arr, int low1, int high1, int low2, int high2, int *B, int lowB);
int find_split(int *arr, int low, int high, int x);
size_t binary_search(int *arr, size_t low, size_t high, int x);

void msort_multip(int *arr, size_t low, size_t high, int *B, size_t threshold) {
    print_mutex.lock();
    // std::cout << "msort_multip - " << "low: " << low << " high: " << high << std::endl;
    print_mutex.unlock();

    if (low >= high) return;
    if ((high - low + 1) < threshold) {
        insertion_sort_inp(arr, low, high);
        return;
    }
    size_t mid = (low + high) / 2;
    #pragma omp task
    {
        msort_multip(arr, low, mid, B, threshold);
    }
    #pragma omp task
    {
        msort_multip(arr, mid + 1, high, B, threshold);
    }
    #pragma omp taskwait

    mergep(arr, low, mid, mid + 1, high, B, low);
    for (size_t k = low; k <= high; k++) {
        arr[k] = B[k];
    }
}

/*
Parallel merge
*/
void mergep(int *arr, int low1, int high1, int low2, int high2, int *B, int lowB) {
    print_mutex.lock();
    // std::cout << "mergep - " << "low1: " << low1 << " high1: " << high1 
    //     << " low2: " << low2 << " high2: " << high2
    //     << " lowB: " << lowB << std::endl;
    print_mutex.unlock();

    if (low1 > high1 && low2 > high2) return;

    // if second array bigger, swap
    if ((high1 - low1) < (high2 - low2)) {
        mergep(arr, low2, high2, low1, high1, B, lowB);
        return;
    }

    int mid1 = (low1 + high1) / 2;
    int pos2 = find_split(arr, low2, high2, arr[mid1]);
    int posB = lowB + (mid1 - low1) + (pos2 - low2);
    B[posB] = arr[mid1];
    
    #pragma omp task
    {
        mergep(arr, low1, mid1 - 1, low2, pos2 - 1, B, lowB);
    }
    #pragma omp task
    {
        mergep(arr, mid1 + 1, high1, pos2, high2, B, posB + 1);
    }
    #pragma omp taskwait
}

/*
Find split

Similar to binary search
*/
int find_split(int *arr, int low, int high, int x) {
    high = high + 1;
    while (low < high) {
        int mid = (low + high) / 2;
        if (arr[mid] < x) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

/*
Binary search

If not found, returns 
    - the index of the last element less than x, 
    - or 0 if x is less than all elements
If found, returns the index of the first element equal to x
*/
size_t binary_search(int *arr, size_t low, size_t high, int x) {
    if (low >= high) return low;
    
    size_t mid = (low + high) / 2;
    if (arr[mid] == x) return mid;
    if (arr[mid] < x) return binary_search(arr, mid + 1, high, x);
    return binary_search(arr, low, mid - 1, x);
}

void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    int* B = (int*) malloc (n * sizeof(int));
    msort_multi(arr, 0, n - 1, B, threshold);
}

void arr_print(int* arr, const size_t n);
void arr_init(int* arr, const size_t n);

int mainc(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: ./msort N" << std::endl;
        return 1;
    }
    size_t N = atoi(argv[1]);
    size_t THRES_PRINT = 15;
    size_t thres_elem = 8;

    int arr[N];
    int B[N];
    std::chrono::duration<double, std::milli> duration;

    srand(time(NULL));

    arr_init(arr, N);
    if (N <= THRES_PRINT)
        arr_print(arr, N);
    
    auto start = std::chrono::high_resolution_clock::now();
    msort_single(arr, 0, N-1, B);
    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end - start);
    std::cout << "single thread took: " << duration.count() << "ms" << std::endl;
    if (N <= THRES_PRINT)
        arr_print(arr, N);

    arr_init(arr, N);
    if (N <= THRES_PRINT)
        arr_print(arr, N);
    insertion_sort_inp(arr, 0, N-1);
    std::cout << "insertion sort" << std::endl;
    if (N <= THRES_PRINT)
        arr_print(arr, N);

    arr_init(arr, N);
    if (N <= THRES_PRINT)
        arr_print(arr, N);
    start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(8)
    {
        #pragma omp single
        msort_multi(arr, 0, N-1, B, thres_elem);
    }
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end - start);
    std::cout << "multi thread took: " << duration.count() << "ms" << std::endl;
    if(N <= THRES_PRINT)
        arr_print(arr, N);

    arr_init(arr, N);
    if (N <= THRES_PRINT)
        arr_print(arr, N);
    start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel num_threads(8)
    {
        #pragma omp single
        msort_multip(arr, 0, N - 1, B, thres_elem);
    }
     

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end - start);
    std::cout << "multi-thread parallel merge took: " << duration.count() << "ms" << std::endl;
    if(N <= THRES_PRINT)
        arr_print(arr, N);

    return 0;
}


void insertion_sort_inp(int* arr, int low, int high) {
    // print_mutex.lock();
    // std::cout << "insertion sort inp - " << "low: " << low << " high: " << high << std::endl;   
    // print_mutex.unlock();
    for (int i = low; i <= high; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= low && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

void insertion_sort(const int* arr, int low, int high, int *B) {
    // print_mutex.lock();
    // std::cout << "insertion sort - " << "low: " << low << " high: " << high << std::endl;   
    // print_mutex.unlock();
    for (int i = low; i <= high; i++) {
        B[i] = arr[i];
    }
    for (int i = low; i <= high; i++) {
        int key = B[i];
        int j = i - 1;
        while (j >= low && B[j] > key) {
            B[j + 1] = B[j];
            j--;
        }
        B[j + 1] = key;
    }
}


void arr_print(int* arr, const size_t n) {
    for (size_t i = 0; i < n; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

void arr_init(int* arr, const size_t n) {
    for (size_t i = 0; i < n; i++) {
        arr[i] = rand() % n;
    }
}

/* 
Results:
N = 10, 8 threads
------
ST: 0.0009 ms
MT: 0.73 ms
MT wTh: 0.61 ms
MTP: 1.12 ms
MTP wTh: 0.25 ms

N = 10000
---------
ST: 0.73 ms
MT: 15.2 ms
MT wTh: 4.58 ms
MTP: 256 ms
MTP wTh: 183 ms
*/

