// Congestion- and Delay-Aware (CDA) task scheduler for the LoRR 2026 start-kit.
//
// Scores every candidate (agent i, task j) pair with
//
//     S(i,j) = d(i,j) + lambda * C(i,j) + gamma * R(i,j)
//
// and resolves the assignment with a global greedy matching over the scored
// pairs.  All three terms are in "action steps" (the unit of the start-kit
// heuristic distance), so lambda and gamma are dimensionless.
//
//   d  travel cost: h(agent, first errand) + cached errand-chain length.
//   C  congestion exposure: route length weighted by the smoothed robot density
//      and stall density of the coarse blocks the route crosses, plus the number
//      of other robots already heading to the task's errand blocks.
//   R  execution-delay risk: expected remaining delay of a currently delayed
//      robot, delayed robots blocking the route, and bottleneck exposure scaled
//      by the online-estimated delay rate and duration.
//
// Nothing here reads DelayConfig: the shared environment only exposes the delay
// model *names*, so p(delay) and E[duration] are estimated from the inDelay flags.
//
// Modes (env var LORR_SCHED): DEFAULT (start-kit greedy), S0, S1, S2, S3,
// ADAPT (state-dependent lambda/gamma), BANDIT (discounted UCB over weight arms).
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "SharedEnv.h"

namespace CDA {

struct Config
{
    std::string mode = "DEFAULT";
    double lam = 0.0;          // congestion weight
    double gam = 0.0;          // delay-risk weight
    int shortlist_k = 16;      // candidate tasks kept per free agent (by d)
    double ema = 0.2;          // smoothing of the block fields per scheduler call
    double alpha_stall = 2.0;  // weight of stalled-robot density inside C
    double beta_goal = 1.0;    // steps per competing robot heading to the same block
    double kappa_corr = 1.0;   // bottleneck amplification of a delay inside R
    // adaptive mode references
    double stall_ref = 0.15;   // busy-robot stall fraction at which lambda_t = lambda (tuning-set median)
    double delay_ref = 0.05;   // delayed-robot fraction at which gamma_t = gamma (tuning-set median)
    double adapt_max = 3.0;    // clip of the adaptive multipliers
    // bandit mode
    int bandit_window = 100;   // ticks per arm pull
    double bandit_discount = 0.95;
    double bandit_c = 0.5;
    std::string stats_path;    // JSON with per-call telemetry, written at exit
    int seed = 0;
};

Config config_from_env();

void initialize(SharedEnvironment* env, const Config& cfg);
void plan(int time_limit_ms, std::vector<int>& proposed_schedule, SharedEnvironment* env);
void write_stats();

} // namespace CDA
