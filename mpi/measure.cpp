
#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

/*
an MPI program to quantify the communication latency and bandwidth between two MPI
processes (executed on the same node) using MPI Send and MPI Recv
*/

int main(int argc, char** argv) {
    int rank, size;
    int n;
    float* send_buf;
    float* recv_buf;
    MPI_Status status;
    double t0, t1, t2, t3, t4, t5;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <n>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    n = std::atoi(argv[1]);
    send_buf = (float *) malloc(n * sizeof(float));
    recv_buf = (float *) malloc(n * sizeof(float));

    std::srand(std::time(nullptr));

    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            send_buf[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        }

        t0 = MPI_Wtime();
        MPI_Send(send_buf, n, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(recv_buf, n, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &status);
        t1 = MPI_Wtime();
        
        t4 = t1 - t0;
        MPI_Send(&t4, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        t2 = MPI_Wtime();
        MPI_Recv(recv_buf, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Send(send_buf, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        t3 = MPI_Wtime();
        
        MPI_Recv(&t5, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

        std::cout << (t3 - t2 + t5) * 1000 << std::endl;
    }

    free(send_buf);
    free(recv_buf);

    MPI_Finalize();
    return 0;
}
