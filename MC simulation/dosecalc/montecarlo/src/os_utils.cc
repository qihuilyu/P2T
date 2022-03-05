#include "os_utils.hh"

#include <cstdio>
#include <cstring>

bool file_exists(const std::string& fname) {
    struct stat buf;
    return stat(fname.c_str(), &buf) == 0;
}
int create_directory(const std::string& dir, const bool exist_ok, const mode_t perms) {
#if defined(_WIN32)
    int _fail = _mkdir(dir.c_str());
#else
    int _fail = mkdir(dir.c_str(), perms);
#endif
    if (_fail && (errno!=EEXIST || !exist_ok)) {
        printf("create_directory() failed with error (%d): %s\n", errno, std::strerror(errno));
    }
    return errno;
}
