// transpose_bwd.h
#pragma once
class TransposeBwd {
public:
    TransposeBwd() = default;
    virtual ~TransposeBwd() = default;
    TransposeBwd(const TransposeBwd&) = delete;
    TransposeBwd& operator=(const TransposeBwd&) = delete;
    TransposeBwd(TransposeBwd&&) = delete;
    TransposeBwd& operator=(TransposeBwd&&) = delete;
    virtual void execute() = 0;
};
