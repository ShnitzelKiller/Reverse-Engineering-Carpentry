#pragma once
#include <Eigen/Dense>
#include <fstream>
#include <memory>

static std::unique_ptr<char[]> read_file(const std::string &filename, size_t &size, size_t offset=0, size_t maxSize=0) {
    std::ifstream is(filename, std::ios::binary | std::ios::ate);
    if (!is) {
        size = 0;
        return std::unique_ptr<char[]>(nullptr);
    }
    size = is.tellg();
    size -= offset;
    if (maxSize > 0 && size > maxSize) {
        size = maxSize;
    }
    std::unique_ptr<char[]> memblock(new char[size]);
    is.seekg(offset, std::ios::beg);
    is.read(memblock.get(), size);
    return memblock;
}

template<typename T>
static const char *read_object(const char *buffer, T& target) {
    target = *reinterpret_cast<const T*>(buffer);
    return buffer + sizeof(T);
}

template<typename CT>
static const char *read_object(const char *buffer, std::vector<CT>& target) {
    size_t size = target.size();
    CT const *buf_start = reinterpret_cast<const CT*>(buffer);
    std::copy(buf_start, buf_start + size, target.begin());
    return buffer + size * sizeof(CT);
}

template<typename CT>
static const char *read_object(const char *buffer, Eigen::Quaternion<CT>& target) {
    CT const *buf_start = reinterpret_cast<const CT*>(buffer);
    target = Eigen::Quaternion<CT>(buf_start);
    return buffer + 4 * sizeof(CT);
}

template<typename CT, int N>
static const char *read_object(const char *buffer, Eigen::Matrix<CT, N, 1>& target) {
    CT const *buf_start = reinterpret_cast<const CT*>(buffer);
    target = Eigen::Matrix<CT, N, 1>(buf_start);
    return buffer + N * sizeof(CT);
}

static const char *read_object(const char *buffer, std::string &string, size_t size=0) {
    if (size == 0) {
        string = buffer;
        return buffer + string.size() + 1;
    } else {
        char newstring[size+1];
        memcpy(newstring, buffer, size);
        newstring[size] = '\0';
        string = newstring;
        return buffer + size;
    }
}
