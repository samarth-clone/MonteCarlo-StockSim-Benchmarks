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
#include <random>
#include <chrono>
#include <omp.h>  // OpenMP header
#include "timer_util.h"  // Include our timing utility

namespace fs = std::filesystem;

// Structure to hold stock parameters for simulation
struct StockParams {
    std::string ticker;
    float S0;
    float mu;
    float sigma;
    float dt;
    int steps;
    int paths;
};

// Parse date-price pairs from CSV file
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

void simulateStock(const StockParams& stock, PerformanceTimer& timer) {
    timer.start_timing();
    std::vector<std::vector<float>> results(stock.paths, std::vector<float>(stock.steps));
    
    int num_threads = omp_get_max_threads();
    std::cout << "Running with " << num_threads << " OpenMP threads" << std::endl;
    timer.stop_timing("omp_memory_allocation");
    
    timer.start_timing();
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num() * 123456789;
        std::mt19937 gen(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        
        #pragma omp for
        for (int i = 0; i < stock.paths; ++i) {
            float S = stock.S0;
            for (int t = 0; t < stock.steps; ++t) {
                float Z = normal(gen);
                S *= exp((stock.mu - 0.5f*stock.sigma*stock.sigma)*stock.dt + stock.sigma*sqrt(stock.dt)*Z);
                results[i][t] = S;
            }
        }
    }  
    timer.stop_timing("omp_monte_carlo_simulation");

    timer.start_timing();
    fs::path outDir = "predictions";
    fs::create_directories(outDir);

    fs::path simPath = outDir / ("omp_predicted_traversals_" + stock.ticker + ".csv");
    std::ofstream file(simPath);
    if (!file.is_open()) {
        std::cerr << "Cannot write simulation data to " << simPath << std::endl;
    } else {
        for (int i = 0; i < stock.paths; ++i) {
            for (int j = 0; j < stock.steps; ++j) {
                file << results[i][j] << (j+1<stock.steps ? "," : "");
            }
            file << "\n";
        }
        std::cout << "Saved predicted traversals to " << simPath << std::endl;
    }
    timer.stop_timing("omp_save_traversals");

    timer.start_timing();
    std::vector<double> meanPath(stock.steps, 0.0);
    
    #pragma omp parallel
    {
        std::vector<double> localSum(stock.steps, 0.0);
        
        #pragma omp for
        for (int i = 0; i < stock.paths; ++i) {
            for (int j = 0; j < stock.steps; ++j) {
                localSum[j] += results[i][j];
            }
        }
        
        #pragma omp critical
        {
            for (int j = 0; j < stock.steps; ++j) {
                meanPath[j] += localSum[j];
            }
        }
    }
    
    for (int j = 0; j < stock.steps; ++j) {
        meanPath[j] /= stock.paths;
    }
    timer.stop_timing("omp_compute_mean_path");

    timer.start_timing();
    fs::path predPath = outDir / ("omp_prediction_walk_" + stock.ticker + ".csv");
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
    timer.stop_timing("omp_save_mean_path");
}

int main(int argc, char** argv) {
    std::string ticker = "AAPL";
    if (argc >= 2) {
        ticker = argv[1];
    }

    PerformanceTimer timer("openmp", ticker);
    timer.start_timing();

    std::string dataFile = "historic_data/" + ticker + "_historical.csv";
    if (!fs::exists(dataFile)) {
        std::cout << "Data file not found: " << dataFile << "\n";
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
        1.0f/252.0f,    // dt (daily)
        100,            // steps (next 100 days)
        100000          // paths
    };
    
    std::cout << "Starting simulation with " << stock.paths << " paths over " << stock.steps << " days...\n";
    
    timer.start_timing();
    simulateStock(stock, timer);
    timer.stop_timing("total_simulation_time");
    
    timer.save_to_csv();
    
    std::cout << "Results saved in predictions/ directory." << std::endl;
    std::cout << "Timing information saved to timing_openmp_" << ticker << ".csv" << std::endl;

    return 0;
}