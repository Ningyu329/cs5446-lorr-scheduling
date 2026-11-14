// Replacement for the start-kit src/TaskScheduler.cpp: dispatches to the CDA
// scheduler (sched/cda_scheduler.cpp). With LORR_SCHED unset or DEFAULT the
// behaviour is the start-kit default greedy scheduler.
#include "TaskScheduler.h"

#include "scheduler.h"
#include "const.h"
#include "cda_scheduler.h"

void TaskScheduler::initialize(int preprocess_time_limit)
{
    CDA::initialize(env, CDA::config_from_env());
}

void TaskScheduler::plan(int time_limit, std::vector<int> & proposed_schedule)
{
    // same budget split as the start-kit: at most half of the entry time limit
    int limit = time_limit/2 - DefaultPlanner::SCHEDULER_TIMELIMIT_TOLERANCE;
    CDA::plan(limit, proposed_schedule, env);
}
