#include <iostream>
#include "Profiler.h"

using namespace std;

CTimeProfiler::CTimeProfiler(char const* const name)
    : m_name(name)
    , m_startTime(clock())
{
}

CTimeProfiler::~CTimeProfiler()
{
    double const executionTime = 1000.0 * ((double)(std::clock() - m_startTime) / (double)CLOCKS_PER_SEC);
    cerr.precision(3);
    cerr /*<< "Execution time "*/ << m_name << ": " << executionTime << fixed << " ms" << endl;
}
