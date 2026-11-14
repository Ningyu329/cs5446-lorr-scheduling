#include "cda_scheduler.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <numeric>
#include <unordered_set>

#include "heuristics.h"
#include "scheduler.h"
#include "nlohmann/json.hpp"

namespace CDA {

using Clock = std::chrono::steady_clock;

namespace {

// ---------------------------------------------------------------------------
// persistent scheduler state
// ---------------------------------------------------------------------------
Config cfg;
SharedEnvironment* E = nullptr;

int rows = 0, cols = 0, B = 1, BR = 1, BC = 1, NB = 1;
std::vector<int> free_cells;      // passable cells per block
std::vector<double> corr;         // share of corridor cells (<=2 passable neighbours)
std::vector<double> occ_ema;      // smoothed robots per block
std::vector<double> stall_ema;    // smoothed stalled busy robots per block
std::vector<int> delay_now;       // robots currently delayed per block
std::vector<int> goal_cnt;        // robots whose next errand lies in the block
std::vector<double> cong_field;   // per-call C density, see refresh_fields
std::vector<double> haz_field;    // per-call R bottleneck hazard

std::vector<int> prev_loc, prev_ori, delay_since;
std::vector<char> prev_delay;
int prev_ts = -1;

// online delay model (ticks)
double exposure_ticks = 0.0;  // sum over calls of (#undelayed robots * dt)
long long onsets = 0;
double episode_sum = 0.0;
long long episodes = 0;
double stall_frac = 0.0, delay_frac = 0.0;

// task bookkeeping
std::unordered_map<int, std::pair<int, std::vector<int>>> task_chain;  // id -> (total, segment lens)
std::unordered_map<int, std::pair<int, int>> task_progress;           // id -> (idx_next_loc, n_errands)
long long errands_done = 0;

// bandit over (lambda, gamma) multipliers
std::vector<std::pair<double, double>> arms;
std::vector<double> arm_n, arm_x;
int arm = -1, window_start = 0;
long long window_errands = 0;
std::vector<int> arm_pulls;

// telemetry
std::mutex stats_mu;
nlohmann::json calls = nlohmann::json::array();
nlohmann::json assigns = nlohmann::json::array();
int last_write_ts = 0;
int horizon = 0;  // LORR_HORIZON: the start-kit exits via _exit, so the last write must precede it

bool due_write(int ts)
{
    return ts - last_write_ts >= 250 || (horizon > 0 && ts >= horizon - 25);
}

double env_d(const char* k, double def)
{
    const char* v = std::getenv(k);
    return v ? std::atof(v) : def;
}

int block_of(int loc)
{
    int r = loc / cols, c = loc % cols;
    return (r / B) * BC + (c / B);
}

double p_hat()   { return exposure_ticks > 0 ? onsets / exposure_ticks : 0.0; }
double dur_hat() { return episodes > 0 ? episode_sum / episodes : 0.0; }

// ---------------------------------------------------------------------------
// observation update: robot density, stalls, delays, errand progress
// ---------------------------------------------------------------------------
void observe()
{
    const int n = E->num_of_agents;
    const int ts = E->system_timestep;
    const int dt = prev_ts < 0 ? 0 : std::max(0, ts - prev_ts);

    std::vector<int> occ(NB, 0), stall(NB, 0);
    std::fill(delay_now.begin(), delay_now.end(), 0);
    std::fill(goal_cnt.begin(), goal_cnt.end(), 0);

    int busy = 0, stalled = 0, delayed = 0, undelayed = 0;
    for (int i = 0; i < n; i++)
    {
        const State& s = E->system_states[i];  // actual, not the predicted planning start
        if (s.location < 0)
            continue;
        int b = block_of(s.location);
        occ[b]++;
        bool in_delay = s.delay.inDelay;
        if (in_delay)
        {
            delayed++;
            delay_now[b]++;
            if (!prev_delay[i])
            {
                onsets++;
                delay_since[i] = ts;
            }
        }
        else
        {
            undelayed++;
            if (prev_delay[i] && delay_since[i] >= 0)
            {
                episode_sum += std::max(1, ts - delay_since[i]);
                episodes++;
                delay_since[i] = -1;
            }
        }
        int t_id = E->curr_task_schedule.size() > (size_t)i ? E->curr_task_schedule[i] : -1;
        if (t_id >= 0)
        {
            busy++;
            auto it = E->task_pool.find(t_id);
            if (it != E->task_pool.end() && it->second.idx_next_loc < (int)it->second.locations.size())
                goal_cnt[block_of(it->second.locations[it->second.idx_next_loc])]++;
            if (!in_delay && dt >= E->max_counter && prev_loc[i] == s.location && prev_ori[i] == s.orientation)
            {
                stalled++;
                stall[b]++;
            }
        }
        prev_loc[i] = s.location;
        prev_ori[i] = s.orientation;
        prev_delay[i] = in_delay;
    }
    if (dt > 0)
        exposure_ticks += (double)undelayed * dt;

    double a = cfg.ema;
    for (int b = 0; b < NB; b++)
    {
        occ_ema[b] = (1 - a) * occ_ema[b] + a * occ[b];
        stall_ema[b] = (1 - a) * stall_ema[b] + a * stall[b];
    }
    stall_frac = (1 - a) * stall_frac + a * (busy > 0 ? (double)stalled / busy : 0.0);
    delay_frac = (1 - a) * delay_frac + a * (n > 0 ? (double)delayed / n : 0.0);

    // errand progress, including tasks that left the pool since the last call
    std::unordered_set<int> seen;
    for (auto& kv : E->task_pool)
    {
        seen.insert(kv.first);
        auto it = task_progress.find(kv.first);
        int idx = kv.second.idx_next_loc;
        if (it == task_progress.end())
            task_progress[kv.first] = {idx, (int)kv.second.locations.size()};
        else
        {
            errands_done += std::max(0, idx - it->second.first);
            it->second.first = idx;
        }
    }
    for (auto it = task_progress.begin(); it != task_progress.end();)
    {
        if (!seen.count(it->first))
        {
            errands_done += std::max(0, it->second.second - it->second.first);
            task_chain.erase(it->first);
            it = task_progress.erase(it);
        }
        else
            ++it;
    }
    prev_ts = ts;
}

void refresh_fields()
{
    for (int b = 0; b < NB; b++)
    {
        double fc = std::max(1, free_cells[b]);
        double rho = std::min(1.0, occ_ema[b] / fc);
        double sig = std::min(1.0, stall_ema[b] / fc);
        cong_field[b] = rho + cfg.alpha_stall * sig;
        haz_field[b] = cfg.kappa_corr * corr[b] + rho;
    }
}

const std::pair<int, std::vector<int>>& chain_of(int t_id)
{
    auto it = task_chain.find(t_id);
    if (it != task_chain.end())
        return it->second;
    const Task& t = E->task_pool.at(t_id);
    std::vector<int> segs;
    int tot = 0;
    for (size_t k = t.idx_next_loc + 1; k < t.locations.size(); k++)
    {
        int h = DefaultPlanner::get_h(E, t.locations[k - 1], t.locations[k]);
        segs.push_back(h);
        tot += h;
    }
    return task_chain.emplace(t_id, std::make_pair(tot, std::move(segs))).first->second;
}

// mean of a block field along the straight segment a->b, plus a sum of delayed robots
void sample_segment(int a, int b, double& mean_c, double& mean_h, double& dly)
{
    int ra = a / cols, ca = a % cols, rb = b / cols, cb = b % cols;
    int span = std::abs(ra - rb) + std::abs(ca - cb);
    int ns = std::max(1, span / B) + 1;
    double sc = 0, sh = 0;
    int last = -1;
    for (int k = 0; k < ns; k++)
    {
        double f = ns == 1 ? 0.0 : (double)k / (ns - 1);
        int r = (int)std::lround(ra + f * (rb - ra)), c = (int)std::lround(ca + f * (cb - ca));
        int blk = (r / B) * BC + (c / B);
        sc += cong_field[blk];
        sh += haz_field[blk];
        if (blk != last)
            dly += delay_now[blk];
        last = blk;
    }
    mean_c = sc / ns;
    mean_h = sh / ns;
}

struct Cand
{
    int agent, task;
    int d;
    double C, R, S;
};

// C and R for agent i taking task t whose first leg costs h0 steps
void score(int i, int t_id, int h0, Cand& c, double lam, double gam)
{
    c.C = 0;
    c.R = 0;
    if (lam > 0 || gam > 0)
    {
        const Task& t = E->task_pool.at(t_id);
        const auto& ch = chain_of(t_id);
        int ctr = std::max(1, E->max_counter);
        double dur_steps = dur_hat() / ctr;
        double route_c = 0, route_h = 0, dly = 0;
        int from = E->curr_states[i].location;
        for (size_t k = t.idx_next_loc; k < t.locations.size(); k++)
        {
            int seg = (k == (size_t)t.idx_next_loc) ? h0 : ch.second[k - t.idx_next_loc - 1];
            double mc, mh;
            sample_segment(from, t.locations[k], mc, mh, dly);
            route_c += seg * mc;
            route_h += seg * mh;
            from = t.locations[k];
        }
        double crowd = 0;
        for (size_t k = t.idx_next_loc; k < t.locations.size(); k++)
            crowd += goal_cnt[block_of(t.locations[k])];
        c.C = route_c + cfg.beta_goal * crowd;

        double rem = 0;
        if (E->system_states[i].delay.inDelay && delay_since[i] >= 0)
        {
            double el = E->system_timestep - delay_since[i];
            rem = std::max(dur_hat() - el, 0.5 * dur_hat()) / ctr;
        }
        double ph_steps = p_hat() * ctr;  // delay onsets per action step
        c.R = rem + ph_steps * dur_steps * route_h + 0.5 * dur_steps * dly / B;
    }
    c.S = c.d + lam * c.C + gam * c.R;
}

void current_weights(double& lam, double& gam)
{
    lam = cfg.lam;
    gam = cfg.gam;
    if (cfg.mode == "S0")
        lam = gam = 0;
    else if (cfg.mode == "S1")
        gam = 0;
    else if (cfg.mode == "S2")
        lam = 0;
    else if (cfg.mode == "ADAPT")
    {
        lam = cfg.lam * std::min(cfg.adapt_max, stall_frac / cfg.stall_ref);
        gam = cfg.gam * std::min(cfg.adapt_max, delay_frac / cfg.delay_ref);
    }
    else if (cfg.mode == "BANDIT")
    {
        int ts = E->system_timestep;
        if (arm < 0 || ts - window_start >= cfg.bandit_window)
        {
            if (arm >= 0)
            {
                double span = std::max(1, ts - window_start);
                double r = (errands_done - window_errands) / span / std::max(1, E->num_of_agents) * 100.0;
                for (size_t a = 0; a < arms.size(); a++)
                {
                    arm_n[a] *= cfg.bandit_discount;
                    arm_x[a] *= cfg.bandit_discount;
                }
                arm_n[arm] += 1;
                arm_x[arm] += r;
            }
            double tot = std::accumulate(arm_n.begin(), arm_n.end(), 0.0);
            int best = 0;
            double best_v = -1e18;
            for (size_t a = 0; a < arms.size(); a++)
            {
                double v = arm_n[a] < 1e-9 ? 1e18 - a
                                           : arm_x[a] / arm_n[a] + cfg.bandit_c * std::sqrt(std::log(std::max(1.0, tot)) / arm_n[a]);
                if (v > best_v)
                {
                    best_v = v;
                    best = (int)a;
                }
            }
            arm = best;
            arm_pulls.push_back(arm);
            window_start = ts;
            window_errands = errands_done;
        }
        lam = cfg.lam * arms[arm].first;
        gam = cfg.gam * arms[arm].second;
    }
}

} // namespace

// ---------------------------------------------------------------------------
Config config_from_env()
{
    Config c;
    if (const char* m = std::getenv("LORR_SCHED"))
        c.mode = m;
    c.lam = env_d("LORR_LAMBDA", c.lam);
    c.gam = env_d("LORR_GAMMA", c.gam);
    c.shortlist_k = (int)env_d("LORR_K", c.shortlist_k);
    c.ema = env_d("LORR_EMA", c.ema);
    c.alpha_stall = env_d("LORR_ALPHA", c.alpha_stall);
    c.beta_goal = env_d("LORR_BETA", c.beta_goal);
    c.kappa_corr = env_d("LORR_KAPPA", c.kappa_corr);
    c.stall_ref = env_d("LORR_STALL_REF", c.stall_ref);
    c.delay_ref = env_d("LORR_DELAY_REF", c.delay_ref);
    c.bandit_window = (int)env_d("LORR_BANDIT_WINDOW", c.bandit_window);
    c.seed = (int)env_d("LORR_SEED", 0);
    horizon = (int)env_d("LORR_HORIZON", 0);
    if (const char* p = std::getenv("LORR_STATS"))
        c.stats_path = p;
    return c;
}

void write_stats()
{
    if (cfg.stats_path.empty())
        return;
    std::lock_guard<std::mutex> lk(stats_mu);
    nlohmann::json j;
    j["config"] = {{"mode", cfg.mode}, {"lam", cfg.lam}, {"gam", cfg.gam}, {"K", cfg.shortlist_k},
                   {"ema", cfg.ema}, {"alpha_stall", cfg.alpha_stall}, {"beta_goal", cfg.beta_goal},
                   {"kappa_corr", cfg.kappa_corr}, {"block", B}};
    j["p_hat"] = p_hat();
    j["dur_hat"] = dur_hat();
    j["onsets"] = onsets;
    j["episodes"] = episodes;
    j["errands_done"] = errands_done;
    j["arm_pulls"] = arm_pulls;
    j["arms"] = arms;
    j["calls"] = calls;
    j["assigns"] = assigns;
    std::ofstream f(cfg.stats_path + ".tmp");
    f << j.dump();
    f.close();
    std::rename((cfg.stats_path + ".tmp").c_str(), cfg.stats_path.c_str());
}

void initialize(SharedEnvironment* env, const Config& c)
{
    cfg = c;
    E = env;
    rows = env->rows;
    cols = env->cols;
    B = std::max(2, (int)std::ceil(std::max(rows, cols) / 64.0));
    BR = (rows + B - 1) / B;
    BC = (cols + B - 1) / B;
    NB = BR * BC;
    free_cells.assign(NB, 0);
    std::vector<int> corr_cells(NB, 0);
    for (int loc = 0; loc < rows * cols; loc++)
    {
        if (env->map[loc] != 0)
            continue;
        int b = block_of(loc), r = loc / cols, cc = loc % cols, deg = 0;
        if (r > 0 && env->map[loc - cols] == 0) deg++;
        if (r < rows - 1 && env->map[loc + cols] == 0) deg++;
        if (cc > 0 && env->map[loc - 1] == 0) deg++;
        if (cc < cols - 1 && env->map[loc + 1] == 0) deg++;
        free_cells[b]++;
        if (deg <= 2) corr_cells[b]++;
    }
    corr.assign(NB, 0.0);
    for (int b = 0; b < NB; b++)
        corr[b] = free_cells[b] > 0 ? (double)corr_cells[b] / free_cells[b] : 0.0;
    occ_ema.assign(NB, 0.0);
    stall_ema.assign(NB, 0.0);
    delay_now.assign(NB, 0);
    goal_cnt.assign(NB, 0);
    cong_field.assign(NB, 0.0);
    haz_field.assign(NB, 0.0);
    int n = env->num_of_agents;
    prev_loc.assign(n, -1);
    prev_ori.assign(n, -1);
    prev_delay.assign(n, 0);
    delay_since.assign(n, -1);
    arms = {{0, 0}, {1, 0}, {0, 1}, {0.5, 0.5}, {1, 1}, {2, 2}};
    arm_n.assign(arms.size(), 0.0);
    arm_x.assign(arms.size(), 0.0);
    if (cfg.mode == "DEFAULT")
        DefaultPlanner::schedule_initialize(0, env);
    if (!cfg.stats_path.empty())
        std::atexit(write_stats);
}

void plan(int time_limit_ms, std::vector<int>& proposed_schedule, SharedEnvironment* env)
{
    auto t0 = Clock::now();
    auto endtime = t0 + std::chrono::milliseconds(std::max(1, time_limit_ms));
    E = env;
    if (cfg.mode == "DEFAULT")
    {
        int nf = (int)env->new_freeagents.size();
        DefaultPlanner::schedule_plan(time_limit_ms, proposed_schedule, env);
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        {
            std::lock_guard<std::mutex> lk(stats_mu);
            calls.push_back({env->system_timestep, nf, (int)env->new_tasks.size(), -1, std::round(ms * 100) / 100,
                             0, 0, 0, 0, 0, 0, -1, 0});
        }
        if (due_write(env->system_timestep))
        {
            last_write_ts = env->system_timestep;
            write_stats();
        }
        return;
    }
    observe();
    refresh_fields();
    double lam, gam;
    current_weights(lam, gam);

    const int n = env->num_of_agents;
    // Only entries of free agents are written. The simulator thread keeps editing
    // proposed_schedule while we plan (it resets agents that finish a task), so
    // rewriting busy agents' entries from the planning-start snapshot would revive
    // finished tasks ("task already finished" schedule errors).
    if ((int)proposed_schedule.size() != n)
        proposed_schedule.resize(n, -1);

    std::vector<int> free_agents, free_tasks;
    for (int i = 0; i < n; i++)
        if (env->curr_task_schedule[i] == -1)
            free_agents.push_back(i);
    for (auto& kv : env->task_pool)
        if (kv.second.agent_assigned == -1 && kv.second.idx_next_loc == 0)
            free_tasks.push_back(kv.first);
    std::sort(free_tasks.begin(), free_tasks.end());

    int n_assigned = 0, rounds = 0;
    bool timed_out = false;
    if (!free_agents.empty() && !free_tasks.empty())
    {
        // 1) distance from every free agent to every free task
        std::vector<std::vector<std::pair<int, int>>> dlist(free_agents.size());  // (d, task index)
        size_t n_dist = 0;
        for (size_t a = 0; a < free_agents.size() && !timed_out; a++, n_dist++)
        {
            int loc = env->curr_states[free_agents[a]].location;
            auto& dl = dlist[a];
            dl.reserve(free_tasks.size());
            for (size_t k = 0; k < free_tasks.size(); k++)
            {
                if (k % 32 == 0 && Clock::now() > endtime)
                {
                    timed_out = true;
                    break;
                }
                int t_id = free_tasks[k];
                const Task& t = env->task_pool.at(t_id);
                int h0 = DefaultPlanner::get_h(env, loc, t.locations[t.idx_next_loc]);
                dl.push_back({h0 + chain_of(t_id).first, (int)k});
            }
        }

        std::vector<char> agent_used(free_agents.size(), 0), task_used(free_tasks.size(), 0);
        auto commit = [&](size_t a, const Cand& c, size_t k) {
            proposed_schedule[c.agent] = c.task;
            agent_used[a] = 1;
            task_used[k] = 1;
            n_assigned++;
            if (!cfg.stats_path.empty())
            {
                std::lock_guard<std::mutex> lk(stats_mu);
                assigns.push_back({env->system_timestep, c.agent, c.task, c.d,
                                   std::round(c.C * 100) / 100, std::round(c.R * 100) / 100});
            }
        };

        // 2) rounds of: K nearest *unused* tasks per unmatched agent -> score -> global
        //    greedy on S. Later rounds only re-shortlist agents whose candidates were taken.
        const size_t K = (size_t)std::max(1, cfg.shortlist_k);
        struct Pick { Cand c; size_t a, k; };
        std::vector<std::pair<int, int>> tmp;
        while (rounds < 8)
        {
            std::vector<Pick> cands;
            for (size_t a = 0; a < free_agents.size(); a++)
            {
                if (agent_used[a] || dlist[a].empty())
                    continue;
                tmp.clear();
                for (auto& pr : dlist[a])
                    if (!task_used[pr.second])
                        tmp.push_back(pr);
                size_t kk = std::min(K, tmp.size());
                std::partial_sort(tmp.begin(), tmp.begin() + kk, tmp.end());
                for (size_t q = 0; q < kk; q++)
                {
                    Pick p;
                    p.a = a;
                    p.k = tmp[q].second;
                    p.c.agent = free_agents[a];
                    p.c.task = free_tasks[p.k];
                    p.c.d = tmp[q].first;
                    score(p.c.agent, p.c.task, p.c.d - chain_of(p.c.task).first, p.c, lam, gam);
                    cands.push_back(p);
                }
            }
            if (cands.empty())
                break;
            rounds++;
            std::sort(cands.begin(), cands.end(), [](const Pick& x, const Pick& y) {
                if (x.c.S != y.c.S) return x.c.S < y.c.S;
                if (x.c.agent != y.c.agent) return x.c.agent < y.c.agent;
                return x.c.task < y.c.task;
            });
            for (const Pick& p : cands)
                if (!agent_used[p.a] && !task_used[p.k])
                    commit(p.a, p.c, p.k);
            if (Clock::now() > endtime)
            {
                timed_out = true;
                break;
            }
        }
        // 3) anything still unmatched (round cap / timeout) takes its nearest unused task
        for (size_t a = 0; a < free_agents.size(); a++)
        {
            if (agent_used[a])
                continue;
            int best = -1, best_d = INT_MAX;
            for (auto& pr : dlist[a])
                if (!task_used[pr.second] && pr.first < best_d)
                {
                    best_d = pr.first;
                    best = pr.second;
                }
            if (best >= 0)
                commit(a, Cand{free_agents[a], free_tasks[best], best_d, 0, 0, (double)best_d}, best);
        }
    }

    double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    {
        std::lock_guard<std::mutex> lk(stats_mu);
        calls.push_back({env->system_timestep, (int)free_agents.size(), (int)free_tasks.size(), n_assigned,
                         std::round(ms * 100) / 100, lam, gam, std::round(stall_frac * 1e4) / 1e4,
                         std::round(delay_frac * 1e4) / 1e4, p_hat(), dur_hat(), arm, timed_out ? 1 : 0, rounds});
    }
    if (due_write(env->system_timestep))
    {
        last_write_ts = env->system_timestep;
        write_stats();
    }
}

} // namespace CDA
