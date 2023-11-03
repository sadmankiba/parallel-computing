#include <iostream>
#include <mutex>

#include <omp.h>

unsigned int factorial(unsigned int n) {
    if (n == 0) {
        return 1;
    }
    return n * factorial(n - 1);
}

std::mutex print_mutex;

int main() {
# pragma omp parallel
    {
        #pragma omp single
        {
            std::cout << "Number of threads: " << omp_get_num_threads() << std::endl;
        }
        int tid = omp_get_thread_num();
        
        print_mutex.lock();
        std::cout << "I am thread No. " << tid << std::endl;
        print_mutex.unlock();
        #pragma omp barrier
        
        print_mutex.lock();
        std::cout << (tid * 2 + 1) << "!=" << factorial(tid * 2 + 1) << std::endl;
        print_mutex.unlock();
        print_mutex.lock();
        std::cout << (tid * 2 + 2) << "!=" << factorial(tid * 2 + 2) << std::endl;
        print_mutex.unlock();
    }
}