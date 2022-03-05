#ifndef __OS_UTILS_HH__
#define __OS_UTILS_HH__

#include <string>
#include <sys/stat.h>

bool file_exists(const std::string& fname);
int create_directory(const std::string& dir, const bool exist_ok, const mode_t perms);

#endif // __OS_UTILS_HH__
