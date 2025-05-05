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

void simulateStock(const StockParams& stock, PerformanceTimer& timer) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> normal(0.0f, 1.0f);
    
    std::vector<std::vector<float>> results(stock.paths, std::vector<float>(stock.steps));
    
    timer.start_timing();
    for (int i = 0; i < stock.paths; ++i) {
        float S = stock.S0;
        for (int t = 0; t < stock.steps; ++t) {
            float Z = normal(gen);
            S *= exp((stock.mu - 0.5f*stock.sigma*stock.sigma)*stock.dt + stock.sigma*sqrt(stock.dt)*Z);
            results[i][t] = S;
        }
    }
    timer.stop_timing("monte_carlo_simulation");

    fs::path outDir = "predictions";
    fs::create_directories(outDir);

    fs::path simPath = outDir / ("seq_predicted_traversals_" + stock.ticker + ".csv");
    std::ofstream file(simPath);
    if (!file.is_open()) {
        std::cerr << "Cannot write simulation data to " << simPath << std::endl;
        return;
    } else {
        for (int i = 0; i < stock.paths; ++i) {
            for (int j = 0; j < stock.steps; ++j) {
                file << results[i][j] << (j+1<stock.steps ? "," : "");
            }
            file << "\n";
        }
        std::cout << "Saved predicted traversals to " << simPath << std::endl;
    }

    timer.start_timing();
    std::vector<double> meanPath(stock.steps, 0.0);
    for (int j = 0; j < stock.steps; ++j) {
        double sum = 0.0;
        for (int i = 0; i < stock.paths; ++i) {
            sum += results[i][j];
        }
        meanPath[j] = sum / stock.paths;
    }
    timer.stop_timing("compute_mean_path");

    fs::path predPath = outDir / ("seq_prediction_walk_" + stock.ticker + ".csv");
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
}

int main(int argc, char** argv) {
    std::string ticker = "AAPL";
    if (argc >= 2) {
        ticker = argv[1];
    }

    PerformanceTimer timer("sequential", ticker);
    
    std::string dataFile = "historic_data/" + ticker + "_historical.csv";
    if (!fs::exists(dataFile)) {
        std::cout << "Data file not found: " << dataFile << "\n";
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
        1.0f/252.0f,    // dt (daily)
        100,            // steps (next 100 days)
        100000          // paths (100k for CPU performance)
    };
    
    std::cout << "Starting simulation with " << stock.paths << " paths over " << stock.steps << " days...\n";
    
    simulateStock(stock, timer);
    
    timer.save_to_csv();

    return 0;
}