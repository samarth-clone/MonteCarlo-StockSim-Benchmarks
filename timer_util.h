#ifndef TIMER_UTIL_H
#define TIMER_UTIL_H

#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

struct TimingRecord {
    std::string operation;
    double milliseconds;
};

class PerformanceTimer {
private:
    std::string implementation_name;
    std::string ticker;
    std::vector<TimingRecord> records;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    bool timing_active = false;

public:
    PerformanceTimer(const std::string& impl, const std::string& ticker_symbol) 
        : implementation_name(impl), ticker(ticker_symbol) {}

    void start_timing() {
        start_time = std::chrono::high_resolution_clock::now();
        timing_active = true;
    }

    double stop_timing(const std::string& operation_name) {
        if (!timing_active) {
            std::cerr << "Warning: stop_timing called without active timing" << std::endl;
            return 0.0;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
        double ms = elapsed.count();
        
        records.push_back({operation_name, ms});
        timing_active = false;
        
        std::cout << "[TIMING] " << operation_name << ": " 
                  << std::fixed << std::setprecision(2) << ms << " ms" << std::endl;
        
        return ms;
    }

    void save_to_csv() {
        std::string filename = "timing_" + implementation_name + "_" + ticker + ".csv";
        std::ofstream file(filename);
        
        if (!file.is_open()) {
            std::cerr << "Failed to open file for timing data: " << filename << std::endl;
            return;
        }
        
        file << "Operation,TimeMS,Implementation,Ticker\n";
        
        for (const auto& record : records) {
            file << record.operation << ","
                 << std::fixed << std::setprecision(2) << record.milliseconds << ","
                 << implementation_name << ","
                 << ticker << "\n";
        }
        
        std::cout << "Timing data saved to " << filename << std::endl;
    }
};

#endif 