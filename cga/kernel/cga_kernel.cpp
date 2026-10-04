// Fast cGA kernel (C++), bit-for-bit identical to the Python implementation in cga/simulator.py.
//
// Identity with Python rests on three facts:
//   1. The random numbers are numpy's PCG64 stream (XSL-RR output on a 128-bit LCG), started from
//      the exact state of the numpy Generator, and doubles are made exactly as numpy's
//      Generator.random() does: (next_uint64 >> 11) * 2^-53.
//   2. They are consumed in the same order: per iteration, n doubles for X, then n for Y
//      (numpy's rng.random((2, n)) in row-major order).
//   3. The arithmetic is the same IEEE double arithmetic: p_i +- 1/K, then min(max(p_i, l), u).
//
// The PCG64 outputs are generated on 4 interleaved lanes (jumping 4 steps at a time). This
// yields exactly the same sequence as the serial generator but lets the CPU overlap the 128-bit
// multiplications, which roughly halves the time spent on random numbers.
//
// Build (done automatically by cga/kernel/__init__.py):
//   c++ -O3 -std=c++17 -shared -fPIC -o libcga_kernel.<ext> cga_kernel.cpp

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {

typedef unsigned __int128 u128;

const u128 PCG_MULT = ((u128)0x2360ED051FC65DA4ULL << 64) | 0x4385DF649FCCF645ULL;

inline uint64_t pcg_output(u128 s) {
    uint64_t hi = (uint64_t)(s >> 64), lo = (uint64_t)s, x = hi ^ lo;
    unsigned rot = (unsigned)(hi >> 58);
    return (x >> rot) | (x << ((-rot) & 63));
}

// State after `delta` further steps of the LCG (standard PCG jump-ahead).
u128 pcg_advance(u128 state, u128 inc, uint64_t delta) {
    u128 acc_mult = 1, acc_plus = 0, cur_mult = PCG_MULT, cur_plus = inc;
    while (delta > 0) {
        if (delta & 1) {
            acc_mult *= cur_mult;
            acc_plus = acc_plus * cur_mult + cur_plus;
        }
        cur_plus = (cur_mult + 1) * cur_plus;
        cur_mult *= cur_mult;
        delta >>= 1;
    }
    return acc_mult * state + acc_plus;
}

// Fill out[0..count) (count a multiple of 4) with the next outputs after `state`.
// Returns the state after the last output.
u128 pcg_fill4(u128 state, u128 inc, uint64_t* out, int64_t count) {
    const u128 a2 = PCG_MULT * PCG_MULT, a3 = a2 * PCG_MULT, a4 = a2 * a2;
    const u128 c4 = inc * (a3 + a2 + PCG_MULT + 1);
    u128 s0 = state * PCG_MULT + inc, s1 = s0 * PCG_MULT + inc, s2 = s1 * PCG_MULT + inc,
         s3 = s2 * PCG_MULT + inc;
    for (int64_t i = 0; i < count; i += 4) {
        out[i] = pcg_output(s0);
        out[i + 1] = pcg_output(s1);
        out[i + 2] = pcg_output(s2);
        out[i + 3] = pcg_output(s3);
        if (i + 4 < count) {
            s0 = s0 * a4 + c4;
            s1 = s1 * a4 + c4;
            s2 = s2 * a4 + c4;
            s3 = s3 * a4 + c4;
        }
    }
    return s3;
}

const double TWO_POW_M53 = 1.0 / 9007199254740992.0;

inline bool sample_one(uint64_t raw, double p) { return (double)(raw >> 11) * TWO_POW_M53 < p; }

enum Fitness { BINVAL = 0, ONEMAX = 1 };

}  // namespace

struct CGA {
    int n;
    double step, lo, hi;
    int fitness;
    std::vector<double> p;
    std::vector<int64_t> x, y;  // sampled bits (0/1); 64-bit so the update loop vectorizes with doubles
    // Random numbers: buf[0..buf_len) are the outputs that follow buf_state; pos of them are consumed.
    u128 buf_state, inc;
    std::vector<uint64_t> buf;
    int64_t buf_len, pos;
};

extern "C" {

// Create a kernel. p starts at 1/2. The PCG64 state (state, inc) is numpy's
// Generator.bit_generator.state["state"], split into 64-bit halves.
CGA* cga_create(int n, double step, double lo, double hi, int fitness, uint64_t state_hi,
                uint64_t state_lo, uint64_t inc_hi, uint64_t inc_lo) {
    if (n < 1 || (fitness != BINVAL && fitness != ONEMAX)) return nullptr;
    CGA* c = new CGA();
    c->n = n;
    c->step = step;
    c->lo = lo;
    c->hi = hi;
    c->fitness = fitness;
    c->p.assign(n, 0.5);
    c->x.assign(n, 0);
    c->y.assign(n, 0);
    c->buf_state = ((u128)state_hi << 64) | state_lo;
    c->inc = ((u128)inc_hi << 64) | inc_lo;
    int64_t block = 16 * 2 * (int64_t)n;
    if (block < 8192) block = 8192;
    block = (block + 3) / 4 * 4;
    c->buf.assign(block, 0);
    c->buf_len = 0;  // empty: filled on first use
    c->pos = 0;
    return c;
}

void cga_destroy(CGA* c) { delete c; }

double* cga_p(CGA* c) { return c->p.data(); }

// The serial PCG64 state after the last consumed random number (to continue the stream in numpy).
void cga_rng_state(CGA* c, uint64_t* out4) {
    u128 s = pcg_advance(c->buf_state, c->inc, (uint64_t)c->pos);
    out4[0] = (uint64_t)(s >> 64);
    out4[1] = (uint64_t)s;
    out4[2] = (uint64_t)(c->inc >> 64);
    out4[3] = (uint64_t)c->inc;
}

// Run up to max_iters iterations. Returns the number of iterations executed. If the optimum is
// sampled, *found = 1 and that iteration is the last one executed (counted, no update applied),
// exactly like run_cga in Python.
int64_t cga_run(CGA* c, int64_t max_iters, int32_t* found) {
    const int n = c->n;
    const int64_t need = 2 * (int64_t)n;
    double* p = c->p.data();
    int64_t* x = c->x.data();
    int64_t* y = c->y.data();
    const double step = c->step, lo = c->lo, hi = c->hi;
    *found = 0;
    for (int64_t t = 1; t <= max_iters; t++) {
        if (c->buf_len - c->pos < need) {
            c->buf_state = pcg_advance(c->buf_state, c->inc, (uint64_t)c->pos);
            c->buf_len = (int64_t)c->buf.size();
            pcg_fill4(c->buf_state, c->inc, c->buf.data(), c->buf_len);
            c->pos = 0;
        }
        const uint64_t* u = c->buf.data() + c->pos;
        c->pos += need;

        // Sample X and Y (branch-free, vectorizable).
        int64_t ones_x = 0, ones_y = 0;
        for (int i = 0; i < n; i++) {
            const int64_t xi = sample_one(u[i], p[i]);
            const int64_t yi = sample_one(u[n + i], p[i]);
            x[i] = xi;
            y[i] = yi;
            ones_x += xi;
            ones_y += yi;
        }
        if (ones_x == n || ones_y == n) {  // X or Y is the optimum: stop, no update
            *found = 1;
            return t;
        }
        int first_diff = 0;
        while (first_diff < n && x[first_diff] == y[first_diff]) first_diff++;
        if (first_diff == n) continue;  // X == Y: no update (a tie for every fitness function)
        bool x_wins;
        if (c->fitness == BINVAL) {
            x_wins = x[first_diff];  // the first differing bit decides
        } else {
            if (ones_x == ones_y) continue;  // OneMax tie: no update
            x_wins = ones_x > ones_y;
        }
        // Exactly Python's p += step * (W - L) for every bit (+0.0 where they agree), then clip every bit.
        const int64_t sign = x_wins ? 1 : -1;  // W - L = sign * (X - Y)
        for (int i = 0; i < n; i++) {
            double v = p[i] + step * (double)(sign * (x[i] - y[i]));
            v = v > lo ? v : lo;  // numpy clip: min(max(v, lo), hi)
            v = v < hi ? v : hi;
            p[i] = v;
        }
    }
    return max_iters;
}

}  // extern "C"
