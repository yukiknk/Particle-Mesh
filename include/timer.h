#pragma once
#include <mpi.h>
#include <ctime>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>

class Timer {
public:
    // stat: "avg"（既定。従来どおり平均）または "median"（反復ごとの値の中央値。外れ値に強い）
    Timer(int warm_up, int loop, int world_rank, const std::string& stat = "avg")
        : warm_up_(warm_up), loop_(loop), world_rank_(world_rank), stat_(stat)
    {}

    int register_timer(const std::string& name) {
        names_.push_back(name);
        times_.push_back(0.0);
        samples_.emplace_back();
        return static_cast<int>(names_.size() - 1);
    }

    double get_time() {
        struct timespec tv;
        clock_gettime(CLOCK_REALTIME, &tv);
        
        double told = now_time_;
        now_time_ = tv.tv_sec + tv.tv_nsec * 1e-9;
        return now_time_ - told;
    }

    void start(MPI_Comm comm) {
        MPI_Barrier(comm);
        get_time();
    }

    void stop(int id, MPI_Comm comm) {
        double elapsed = get_time();
        
        if (iteration_ >= warm_up_) {
            double global_max = 0.0;
            MPI_Reduce(&elapsed, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            
            if (world_rank_ == 0) {
                times_[id] += global_max;
                // 同じ反復の中で同じ区間が複数回呼ばれる場合は合算する
                auto& v = samples_[id];
                if (last_iter_.size() <= static_cast<size_t>(id)) last_iter_.resize(id + 1, -1);
                if (last_iter_[id] == iteration_ && !v.empty()) v.back() += global_max;
                else v.push_back(global_max);
                last_iter_[id] = iteration_;
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
        const bool med = (stat_ == "median");
        const char* unit = med ? " sec (median)" : " sec (avg)";
        for (size_t i = 0; i < names_.size(); ++i) {
            double val = med ? median(samples_[i]) : times_[i] / loop_;
            total += val;
            std::cout << std::setw(20) << std::left << names_[i] 
                      << ": " << std::setw(12) << std::right << val << unit << std::endl;
        }
        
        std::cout << "-------------------------" << std::endl;
        std::cout << std::setw(20) << std::left << "Total"
                  << ": " << std::setw(12) << std::right << total << unit << std::endl;
        std::cout << "=========================\n" << std::endl;
    }

private:
    static double median(std::vector<double> v) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        size_t n = v.size();
        return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    }

    int warm_up_;
    int loop_;
    int world_rank_;
    int iteration_ = 0;
    double now_time_ = 0.0;
    
    std::vector<std::string> names_;
    std::vector<double> times_;
    std::string stat_;
    std::vector<std::vector<double>> samples_;   // 反復ごとの値（rank 0 のみ）
    std::vector<int> last_iter_;
};
