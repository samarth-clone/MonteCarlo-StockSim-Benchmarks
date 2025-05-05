// simulate_local.cu
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <filesystem>  
#include <algorithm>
#include "timer_util.h"  

namespace fs = std::filesystem;

struct StockParams {
    std::string ticker;
    float S0;
    float mu;
    float sigma;
    float dt;
    int steps;
    int paths;
};

std::vector<float> readStockDataFromCSV(const std::string &filename) {
    
    std::vector<float> prices;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return prices;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        size_t commaPos = line.find(',');
        if (commaPos != std::string::npos) {
            std::string priceStr = line.substr(commaPos + 1);
            try {
                float price = std::stof(priceStr);
                prices.push_back(price);
            } catch (const std::exception& e) {
                std::cerr << "Error parsing price: " << priceStr << " - " << e.what() << std::endl;
            }
        }
    }
    
    
    return prices;
}

void computeParameters(const std::vector<float>& prices, float &S0, float &mu, float &sigma) {
    
    if (prices.size() < 2) {
        S0 = prices.empty() ? 0.0f : prices.back();
        mu = sigma = 0.0f;
        return;
    }
    S0 = prices.back();
    std::vector<float> logReturns;
    logReturns.reserve(prices.size()-1);
    for (size_t i = 1; i < prices.size(); ++i)
        logReturns.push_back(std::log(prices[i]/prices[i-1]));

    float sum = 0.0f;
    for (float r : logReturns) sum += r;
    float avg = sum / logReturns.size();

    float var = 0.0f;
    for (float r : logReturns)
        var += (r-avg)*(r-avg);

    float stddev = std::sqrt(var / logReturns.size());
    mu    = avg   * 252.0f;
    sigma = stddev * std::sqrt(252.0f);
    
}

__global__ void initRNG(curandState *states, unsigned long seed, int paths) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < paths)
        curand_init(seed, id, 0, &states[id]);
}

__global__ void monteCarloGBM(float *d_results, curandState *states,
                              float S0, float mu, float sigma,
                              float dt, int steps, int paths) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= paths) return;

    curandState localState = states[id];
    float S = S0;
    for (int t = 0; t < steps; ++t) {
        float Z = curand_normal(&localState);
        S *= expf((mu - 0.5f*sigma*sigma)*dt + sigma*sqrtf(dt)*Z);
        d_results[id*steps + t] = S;
    }
    states[id] = localState;
}

__global__ void computeMeanPath(float *d_results, double *meanPath,
                              int steps, int paths) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= steps) return;

    double sum = 0.0;
    for (int i = 0; i < paths; ++i) {
        sum += d_results[i*steps + id];
    }
    meanPath[id] = sum / paths;
}

void checkCudaError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void simulateStock(const StockParams& stock, PerformanceTimer& timer) {
    int total = stock.steps * stock.paths;
    std::vector<float> h_results(total);
    float *d_results;
    curandState *d_states;

    cudaMalloc(&d_results, sizeof(float)*total);
    cudaMalloc(&d_states,  sizeof(curandState)*stock.paths);

    int threads = 256;
    int blocks  = (stock.paths + threads - 1) / threads;

    initRNG<<<blocks, threads>>>(d_states, std::time(nullptr), stock.paths);
    cudaDeviceSynchronize();  
    checkCudaError("initRNG");

    timer.start_timing();
    monteCarloGBM<<<blocks, threads>>>(d_results, d_states,
                                     stock.S0, stock.mu, stock.sigma,
                                     stock.dt, stock.steps, stock.paths);
    cudaDeviceSynchronize();  
    checkCudaError("monteCarloGBM");
    timer.stop_timing("monte_carlo_simulation");

    cudaMemcpy(h_results.data(), d_results,
             sizeof(float)*total, cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy");

    fs::path outDir = "predictions";
    fs::create_directories(outDir);

    fs::path simPath = outDir / ("cuda_predicted_traversals_" + stock.ticker + ".csv");
    std::ofstream file(simPath);
    if (!file.is_open()) {
        std::cerr << "Cannot write simulation data to " << simPath << std::endl;
    } else {
        for (int i = 0; i < stock.paths; ++i) {
            for (int j = 0; j < stock.steps; ++j) {
                file << h_results[i*stock.steps + j] << (j+1<stock.steps ? "," : "");
            }
            file << "\n";
        }
        std::cout << "Saved predicted traversals to " << simPath << std::endl;
    }

    timer.start_timing();
    double *d_meanPath;
    cudaMalloc(&d_meanPath, sizeof(double)*stock.steps);
    computeMeanPath<<<blocks, threads>>>(d_results, d_meanPath,
                                         stock.steps, stock.paths);
    cudaDeviceSynchronize();
    checkCudaError("computeMeanPath");
    std::vector<double> h_meanPath(stock.steps);
    cudaMemcpy(h_meanPath.data(), d_meanPath,
               sizeof(double)*stock.steps, cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy_meanPath");
    cudaFree(d_meanPath);
    checkCudaError("cudaFree_meanPath");
    timer.stop_timing("compute_mean_path");

    fs::path predPath = outDir / ("cuda_prediction_walk_" + stock.ticker + ".csv");
    std::ofstream pf(predPath);
    if (!pf.is_open()) {
        std::cerr << "Cannot write prediction walk to " << predPath << std::endl;
    } else {
        pf << "Day,MeanPrice\n";
        for (int j = 0; j < stock.steps; ++j) {
            pf << (j+1) << "," << h_meanPath[j] << "\n";
        }
        std::cout << "Saved mean walk prediction to " << predPath << std::endl;
    }

    cudaFree(d_results);
    cudaFree(d_states);
}

int main(int argc, char** argv) {
    std::string ticker = "AAPL";
    if (argc >= 2) {
        ticker = argv[1];
    }

    PerformanceTimer timer("cuda", ticker);

    std::string dataFile = "historic_data/" + ticker + "_historical.csv";
    if (!fs::exists(dataFile)) {
        std::cout << "Data file not found\n";
        timer.stop_timing("check_data_file");
        return 1;
    }

    std::cout << "Reading data from " << dataFile << "...\n";
    auto prices = readStockDataFromCSV(dataFile);
    
    if (prices.empty()) {
        std::cerr << "No data read; exiting.\n";
        return 1;
    }
    
    std::cout << "Read " << prices.size() << " price points.\n";

    float S0, mu, sigma;
    computeParameters(prices, S0, mu, sigma);
    std::cout << "Computed S0=" << S0 << " mu=" << mu << " sigma=" << sigma << "\n";


    StockParams stock {
        ticker, S0, mu, sigma,
        1.0f/252.0f,
        100,
        100000
    };

    std::cout << "Starting simulation with " << stock.paths << " paths over " << stock.steps << " days...\n";
    
    simulateStock(stock, timer);
    
    timer.save_to_csv();
    
    std::cout << "Results saved in predictions/ directory." << std::endl;
    std::cout << "Timing information saved to timing_cuda_" << ticker << ".csv" << std::endl;

    return 0;
}