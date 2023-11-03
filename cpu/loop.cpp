#include <cstdio>
#include <iostream>

int main(int argc, char *argv[]){
    if (argc != 2){
        return 1;
    }

    int N = std::atoi(argv[1]);
    for (int i=0; i<N; i++){
        printf("%d ", i);
    }
    printf("%d\n", N);

    for (int i=N; i>0; i--){
        std::cout << i << " ";
    }
    std::cout << "0\n";
    return 0;
}