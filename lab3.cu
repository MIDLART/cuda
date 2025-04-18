#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <cfloat>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

__constant__ float dev_avg[32][3];

__global__ void kernel(uchar4* im, unsigned char* out, int w, int h, int nc) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    for (int x = idx; x < w * h; x += offset) {
        uchar4 pixel = im[x];
        float3 p = make_float3(pixel.x, pixel.y, pixel.z);

        float max_cos = -1;
        int best_class = 0;

        for (int c = 0; c < nc; ++c) {
            float3 class_vec = make_float3(dev_avg[c][0], dev_avg[c][1], dev_avg[c][2]);
            float class_len = sqrtf(class_vec.x * class_vec.x + class_vec.y * class_vec.y + class_vec.z * class_vec.z);

            if (class_len > 1e-6f) {
                class_vec.x /= class_len;
                class_vec.y /= class_len;
                class_vec.z /= class_len;
            }

            float cos_angle = p.x * class_vec.x + p.y * class_vec.y + p.z * class_vec.z;

            if (cos_angle > max_cos) {
                max_cos = cos_angle;
                best_class = c;
            }
        }

        out[x] = (unsigned char)best_class;
    }
}

int main() {
    std::string in, out;

    std::cin >> in >> out;

    int w, h;
    FILE *fp = fopen(in.c_str(), "rb");
    if (!fp) {
        std::cerr << "Failed to open input file" << std::endl;
        return 1;
    }

    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    int nc, np, x, y;
    std::cin >> nc;

    float avg[32][3] = {{0}};

    for (int i = 0; i < nc; ++i) {
        std::cin >> np;
        avg[i][0] = avg[i][1] = avg[i][2] = 0.0f;

        for (int j = 0; j < np; ++j) {
            std::cin >> x >> y;
            if (x >= 0 && x < w && y >= 0 && y < h) {
                uchar4 p = data[y * w + x];
                avg[i][0] += p.x;
                avg[i][1] += p.y;
                avg[i][2] += p.z;
            }
        }

        avg[i][0] /= np;
        avg[i][1] /= np;
        avg[i][2] /= np;
    }

    CSC(cudaMemcpyToSymbol(dev_avg, avg, sizeof(float) * 32 * 3));

    uchar4 *dev_data;
    unsigned char *dev_result;
    CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w * h));
    CSC(cudaMalloc(&dev_result, sizeof(unsigned char) * w * h));
    CSC(cudaMemcpy(dev_data, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    kernel<<<32, 32>>>(dev_data, dev_result, w, h, nc);
    CSC(cudaGetLastError());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

//    std::cout << "Time: " << elapsedTime << " ms" << std::endl;

    unsigned char* result = (unsigned char*)malloc(w * h);
    CSC(cudaMemcpy(result, dev_result, sizeof(unsigned char) * w * h, cudaMemcpyDeviceToHost));

    for (int i = 0; i < w * h; i++) {
        data[i].w = result[i];
    }

    fp = fopen(out.c_str(), "wb");
    if (!fp) {
        std::cerr << "Failed to open output file" << std::endl;
        free(data);
        free(result);
        CSC(cudaFree(dev_data));
        CSC(cudaFree(dev_result));
        return 1;
    }

    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    free(result);
    CSC(cudaFree(dev_data));
    CSC(cudaFree(dev_result));

    return 0;
}