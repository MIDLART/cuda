#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <chrono>

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int x, y;

    for (y = idy; y < h; y += offsety) {
        for (x = idx; x < w; x += offsetx) {
            uchar4 p[3][3];
            float wm[3][3];

            for (long i = 0; i < 3; ++i) {
                for (long j = 0; j < 3; ++j) {

                    float curX = x + j - 1;
                    float curY = y + i - 1;

                    p[i][j] = tex2D<uchar4>(tex, curX, curY);

                    wm[i][j] = 0.299 * p[i][j].x + 0.587 * p[i][j].y + 0.114 * p[i][j].z;
                }
            }

            float Gx = wm[0][2] + 2 * wm[1][2] + wm[2][2] - wm[0][0] - 2 * wm[1][0] - wm[2][0];
            float Gy = wm[2][0] + 2 * wm[2][1] + wm[2][2] - wm[0][0] - 2 * wm[0][1] - wm[0][2];

            float g = sqrtf(Gx * Gx + Gy * Gy);
            g = fminf(g, 255.0f);

            out[y * w + x] = make_uchar4(
                    static_cast<unsigned char>(g),
                    static_cast<unsigned char>(g),
                    static_cast<unsigned char>(g), 255);
        }
    }
}

int main() {
    std::string in, out;

    std::cin >> in >> out;

    int w, h;
    FILE *fp = fopen(in.c_str(), "rb");
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4 *dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    kernel<<< dim3(32, 32), dim3(32, 32) >>>(tex, dev_out, w, h);
    CSC(cudaGetLastError());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

//    std::cout << "Time: " << elapsedTime << " ms" << std::endl;

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    fp = fopen(out.c_str(), "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    return 0;
}