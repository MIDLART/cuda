#include <iostream>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <random>

void fill_random_array(double* arr, int n, double min_val, double max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(min_val, max_val);

    for (int i = 0; i < n; ++i) {
        arr[i] = distrib(gen);
    }
}

__global__ void min_elements (double *arr_1, double *arr_2, double *res, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;

    while (idx < n) {
        if (arr_1[idx] < arr_2[idx]) {
            res[idx] = arr_1[idx];
        } else {
            res[idx] = arr_2[idx];
        }

        idx += offset;
    }
}

void min_elements_cpu (double *arr_1, double *arr_2, double *res, int n) {
    for (int idx = 0; idx < n; ++idx) {
        if (arr_1[idx] < arr_2[idx]) {
            res[idx] = arr_1[idx];
        } else {
            res[idx] = arr_2[idx];
        }
    }
}

int main () {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cout << std::fixed;
    std::cout << std::setprecision(10);

    int n;
//    n = 1000;

    std::cin >> n;

    if (n < 0) {
        return 0;
    }

    double *arr_1 = new double[n];
    double *arr_2 = new double[n];
    double *res = new double[n];

    for (int i = 0; i < n; ++i) {
        std::cin >> arr_1[i];
    }

    for (int i = 0; i < n; ++i) {
        std::cin >> arr_2[i];
    }

//    fill_random_array(arr_1, n, -1000, 1000);
//    fill_random_array(arr_2, n, -1000, 1000);

    double *dev_arr_1, *dev_arr_2, *dev_res;
    cudaMalloc((void**)&dev_arr_1, sizeof(double) * n);
    cudaMalloc((void**)&dev_arr_2, sizeof(double) * n);
    cudaMalloc((void**)&dev_res, sizeof(double) * n);

    cudaMemcpy(dev_arr_1, arr_1, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_arr_2, arr_2, sizeof(double) * n, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    min_elements<<<1, 32>>>(dev_arr_1, dev_arr_2, dev_res, n);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

//    std::cout << "Time: " << elapsedTime << " ms" << std::endl;

    cudaMemcpy(res, dev_res, sizeof(double) * n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        std::cout << res[i] << " ";
    }

    delete [] arr_1;
    delete [] arr_2;
    delete [] res;
    cudaFree(dev_arr_1);
    cudaFree(dev_arr_2);
    cudaFree(dev_res);

    return 0;
}