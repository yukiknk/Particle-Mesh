#pragma once
#include <cstddef>

class TransposeFwd {
public:
    TransposeFwd() = default; 
    virtual ~TransposeFwd() = default;

    TransposeFwd(const TransposeFwd&) = delete;
    TransposeFwd& operator=(const TransposeFwd&) = delete;
    TransposeFwd(TransposeFwd&&) = delete;
    TransposeFwd& operator=(TransposeFwd&&) = delete;
    
    virtual void execute() = 0;
};
