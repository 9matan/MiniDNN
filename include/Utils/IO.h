#pragma once

#ifdef __linux__
#include <sys/types.h>
#include <sys/stat.h>
#elif _WIN32 // #ifdef __linux__
#include <Windows.h>
#endif // #elif _WIN32

namespace MiniDNN
{

void CreateFolder(std::string const& folderName)
{
#ifdef __linux__
    mkdir(folderName.c_str(), ACCESSPERMS);
#elif _WIN32 // #ifdef __linux__
    CreateDirectoryA(folderName.c_str(), NULL);
#endif // #elif _WIN32
}

} // namespace MiniDNN