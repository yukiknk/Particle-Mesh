#pragma once
#include <mpi.h>
#include <ctime>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

class Timer {
public:
    Timer(int warm_up, int loop, int world_rank)
        : warm_up_(warm_up), loop_(loop), world_rank_(world_rank)
    {}

    int register_timer(const std::string& name) {
        names_.push_back(name);
        times_.push_back(0.0);
        return static_cast<int>(names_.size() - 1);
    }

    double get_time() {
        struct timespec tv;
        clock_gettime(CLOCK_REALTIME, &tv);
        
        double told = now_time_;
        now_time_ = tv.tv_sec + tv.tv_nsec * 1e-9;
        return now_time_ - told;
    }

    void start() {
        get_time();
    }

    void stop(int id) {
        double elapsed = get_time();
        
        if (iteration_ >= warm_up_) {
            double global_max = 0.0;
            MPI_Reduce(&elapsed, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            
            if (world_rank_ == 0) {
                times_[id] += global_max;
            }
        }
    }

    void set_iteration(int i) {
        iteration_ = i;
    }

    void print() const {
        if (world_rank_ != 0) return;

        std::cout << "\n===== Timer Results =====" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        
        double total = 0.0;
        for (size_t i = 0; i < names_.size(); ++i) {
            double avg = times_[i] / loop_;
            total += avg;
            std::cout << std::setw(20) << std::left << names_[i] 
                      << ": " << std::setw(12) << std::right << avg << " sec (avg)" << std::endl;
        }
        
        std::cout << "-------------------------" << std::endl;
        std::cout << std::setw(20) << std::left << "Total"
                  << ": " << std::setw(12) << std::right << total << " sec (avg)" << std::endl;
        std::cout << "=========================\n" << std::endl;
    }

private:
    int warm_up_;
    int loop_;
    int world_rank_;
    int iteration_ = 0;
    double now_time_ = 0.0;
    
    std::vector<std::string> names_;
    std::vector<double> times_;
};
