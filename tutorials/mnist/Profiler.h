#pragma once

#include <ctime>
#include <string>

class CTimeProfiler
{
public:
    CTimeProfiler(char const* const name);
    ~CTimeProfiler();
private:
    std::string m_name;
    clock_t m_startTime;
};

#ifndef DISABLE_PROFILE
#define PROFILE_TIME(name) CTimeProfiler timeProfiler(name);
#else
#define PROFILE_TIME(name)
#endif // #ifdef DISABLE_PROFILE