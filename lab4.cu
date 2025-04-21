#include <iostream>
#include <iomanip>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <cmath>
#include <chrono>
#include <random>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

struct comparator {
    __host__ __device__ bool operator()(double a, double b) {
        return fabs(a) < fabs(b);
    }
};

__global__ void swap_rows(double* matrix, int n, int cur_row, int p_row) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += offset) {
        double tmp = matrix[i * n + cur_row];
        matrix[i * n + cur_row] = matrix[i * n + p_row];
        
        matrix[i * n + p_row] = tmp;
    }
}

__global__ void eliminate(double* matrix, int n, int cur_row) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = gridDim.x * blockDim.x;
    int offsety = gridDim.y * blockDim.y;

    for (int i = cur_row + 1 + idx; i < n; i += offsetx) {
        double k = matrix[cur_row * n + i] / matrix[cur_row * n + cur_row];

        for (int j = cur_row + 1 + idy; j < n; j += offsety) {
            matrix[j * n + i] -= k * matrix[j * n + cur_row];
        }
    }
}

int Gauss_direct_course(double* dev_matrix, int n, thrust::device_ptr<double> p_matrix, comparator comp) {
    int swap_count = 0;

    for (int i = 0; i < n - 1; i++) {
        thrust::device_ptr<double> max_el = thrust::max_element(p_matrix + i * n + i, p_matrix + (i + 1) * n, comp);
        int p_row = max_el - (p_matrix + i * n);

        if (p_row != i) {
            swap_rows<<<32, 32>>>(dev_matrix, n, i, p_row);
            CSC(cudaDeviceSynchronize());

            swap_count++;
        }

        double pivot;
        CSC(cudaMemcpy(&pivot, dev_matrix + i * n + i, sizeof(double), cudaMemcpyDeviceToHost));

        if (fabs(pivot) < 1E-7) {
            return -1;
        }

        eliminate<<<dim3(32, 32), dim3(32, 32)>>>(dev_matrix, n, i);
        CSC(cudaDeviceSynchronize());
    }

    return swap_count;
}

double calculate_determinant(double* matrix, int n, int swap_count) {
    double det = 1.0;
    for (int i = 0; i < n; i++) {
        det *= matrix[i * n + i];
    }

    if (swap_count % 2 == 1) {
        det = -det;
    }

    return det;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    int n;
    std::cin >> n;

    double* matrix = new double[n * n];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cin >> matrix[j * n + i];
        }
    }

    double* dev_matrix;
    CSC(cudaMalloc(&dev_matrix, n * n * sizeof(double)));
    CSC(cudaMemcpy(dev_matrix, matrix, n * n * sizeof(double), cudaMemcpyHostToDevice));

    thrust::device_ptr<double> p_matrix = thrust::device_pointer_cast(dev_matrix);
    comparator comp;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int swap_count = Gauss_direct_course(dev_matrix, n, p_matrix, comp);

    if (swap_count == -1) {
        std::cout << std::scientific << std::setprecision(10) << 0.0 << "\n";
    } else {
        CSC(cudaMemcpy(matrix, dev_matrix, n * n * sizeof(double), cudaMemcpyDeviceToHost));

        double det = calculate_determinant(matrix, n, swap_count);
        std::cout << std::scientific << std::setprecision(10) << det << "\n";
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "Time: " << elapsedTime << " ms" << std::endl;

    delete[] matrix;
    CSC(cudaFree(dev_matrix));

    return 0;
}