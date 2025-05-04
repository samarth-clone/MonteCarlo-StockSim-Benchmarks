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

std::vector<float> readStockDataFromCSV(const std::string &filename, PerformanceTimer& timer) {
    timer.start_timing();
    
    std::vector<float> prices;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        timer.stop_timing("file_open_failed");
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
    
    std::reverse(prices.begin(), prices.end());
    
    timer.stop_timing("read_csv_data");
    return prices;
}

void computeParameters(const std::vector<float>& prices, float &S0, float &mu, float &sigma, PerformanceTimer& timer) {
    timer.start_timing();
    
    if (prices.size() < 2) {
        S0 = prices.empty() ? 0.0f : prices.back();
        mu = sigma = 0.0f;
        timer.stop_timing("compute_params_insufficient_data");
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
    
    timer.stop_timing("compute_parameters");
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

void checkCudaError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void simulateStock(const StockParams& stock, PerformanceTimer& timer) {
    timer.start_timing();
    int total = stock.steps * stock.paths;
    std::vector<float> h_results(total);
    float *d_results;
    curandState *d_states;

    cudaMalloc(&d_results, sizeof(float)*total);
    cudaMalloc(&d_states,  sizeof(curandState)*stock.paths);
    timer.stop_timing("cuda_memory_allocation");

    timer.start_timing();
    int threads = 256;
    int blocks  = (stock.paths + threads - 1) / threads;

    initRNG<<<blocks, threads>>>(d_states, std::time(nullptr), stock.paths);
    cudaDeviceSynchronize();  
    checkCudaError("initRNG");
    timer.stop_timing("cuda_rng_initialization");

    timer.start_timing();
    monteCarloGBM<<<blocks, threads>>>(d_results, d_states,
                                     stock.S0, stock.mu, stock.sigma,
                                     stock.dt, stock.steps, stock.paths);
    cudaDeviceSynchronize();  
    checkCudaError("monteCarloGBM");
    timer.stop_timing("cuda_monte_carlo_simulation");

    timer.start_timing();
    cudaMemcpy(h_results.data(), d_results,
             sizeof(float)*total, cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy");
    timer.stop_timing("cuda_copy_results_to_host");

    timer.start_timing();
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
    timer.stop_timing("cuda_save_traversals");

    timer.start_timing();
    std::vector<double> meanPath(stock.steps, 0.0);
    for (int j = 0; j < stock.steps; ++j) {
        double sum = 0.0;
        for (int i = 0; i < stock.paths; ++i) {
            sum += h_results[i * stock.steps + j];
        }
        meanPath[j] = sum / stock.paths;
    }
    timer.stop_timing("cuda_compute_mean_path");

    timer.start_timing();
    fs::path predPath = outDir / ("cuda_prediction_walk_" + stock.ticker + ".csv");
    std::ofstream pf(predPath);
    if (!pf.is_open()) {
        std::cerr << "Cannot write prediction walk to " << predPath << std::endl;
    } else {
        pf << "Day,MeanPrice\n";
        for (int j = 0; j < stock.steps; ++j) {
            pf << (j+1) << "," << meanPath[j] << "\n";
        }
        std::cout << "Saved mean walk prediction to " << predPath << std::endl;
    }
    timer.stop_timing("cuda_save_mean_path");

    timer.start_timing();
    cudaFree(d_results);
    cudaFree(d_states);
    timer.stop_timing("cuda_cleanup");
}

int main(int argc, char** argv) {
    std::string ticker = "AAPL";
    if (argc >= 2) {
        ticker = argv[1];
    }

    PerformanceTimer timer("cuda", ticker);
    timer.start_timing();

    std::string dataFile = "historic_data/" + ticker + "_historical.csv";
    if (!fs::exists(dataFile)) {
        std::cout << "Data file not found\n";
        timer.stop_timing("check_data_file");
        return 1;
    }
    timer.stop_timing("check_data_file");

    std::cout << "Reading data from " << dataFile << "...\n";
    auto prices = readStockDataFromCSV(dataFile, timer);
    
    if (prices.empty()) {
        std::cerr << "No data read; exiting.\n";
        return 1;
    }
    
    std::cout << "Read " << prices.size() << " price points.\n";

    float S0, mu, sigma;
    computeParameters(prices, S0, mu, sigma, timer);
    std::cout << "Computed S0=" << S0 << " mu=" << mu << " sigma=" << sigma << "\n";


    StockParams stock {
        ticker, S0, mu, sigma,
        1.0f/252.0f,
        100,
        100000
    };

    std::cout << "Starting simulation with " << stock.paths << " paths over " << stock.steps << " days...\n";
    
    timer.start_timing();
    simulateStock(stock, timer);
    timer.stop_timing("total_simulation_time");
    
    timer.save_to_csv();
    
    std::cout << "Results saved in predictions/ directory." << std::endl;
    std::cout << "Timing information saved to timing_cuda_" << ticker << ".csv" << std::endl;

    return 0;
}