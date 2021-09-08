#pragma once

#include <vector>
#include <algorithm>
#include <exception>


template <typename K, typename V>
using sorted_map = std::vector<std::pair<K, V>>;

/**
 * Get reference to the value pointed to by given key (throw out of bounds if it doesn't exist)
 */
template <class K, class V>
V& sorted_get(sorted_map<K, V> &map, K key);

/**
 *
 * @tparam K
 * @tparam V
 * @param map
 * @param key
 * @return iterator to found element, or map.end() if not found
 */
template <class K, class V>
typename sorted_map<K, V>::iterator sorted_find(sorted_map<K, V> &map, K key);

/**
 * Insert key value pair into the map non-destructively, returning true if new key was inserted
 */
template <class K, class V>
bool sorted_insert(sorted_map<K, V> &map, K key, V val);

/**
 * Check if sorted map contains a key
 */
template <class K, class V>
bool sorted_contains(sorted_map<K, V> &map, K key);

/**
 * Insert val into vec if not already present, maintaining sorted order
 * @tparam T
 * @param vec sorted vector
 * @param val value to insert
 * @return True if the value was inserted
 */
template <class T>
bool sorted_insert(std::vector<T> &vec, T val);

/**
 * Check whether sorted vector contains a value
 */
template <class T>
bool sorted_contains(std::vector<T> &vec, T val);

/**
 * Find val is in vec in O(log(n)) time.
 * @tparam T
 * @param vec sorted vector
 * @param val value to find
 * @return iterator to found element, or vec.end() if not found
 */
template <class T>
typename std::vector<T>::iterator sorted_find(std::vector<T> &vec, T val);


template <class K, class V>
static bool keycomp(std::pair<K, V> &a, K b) {
    return a.first < b;
}

template <class K, class V>
V& sorted_get(sorted_map<K, V> &map, K key) {
    auto it = std::lower_bound(map.begin(), map.end(), key, keycomp<K, V>);
    if (it == map.end() || key < it->first) {
        throw std::out_of_range("Key not found");
    } else {
        return it->second;
    }
}

template <class K, class V>
typename sorted_map<K, V>::iterator sorted_find(sorted_map<K, V> &map, K key) {
    auto it = std::lower_bound(map.begin(), map.end(), key, keycomp<K, V>);
    if (it != map.end() && key == it->first) {
        return it;
    } else {
        return map.end();
    }
}

template <class K, class V>
bool sorted_insert(sorted_map<K, V> &map, K key, V val) {
    auto it = std::lower_bound(map.begin(), map.end(), key, keycomp<K, V>);
    if (it == map.end() || key < it->first) {
        map.insert(it, std::make_pair(key, val));
        return true;
    } else {
        return false;
    }
}

template <class K, class V>
bool sorted_contains(sorted_map<K, V> &map, K key) {
    auto it = std::lower_bound(map.begin(), map.end(), key, keycomp<K, V>);
    return it != map.end() && it->first == key;
}

template <class T>
bool sorted_insert(std::vector<T> &vec, T val) {
  auto it = std::lower_bound(vec.begin(), vec.end(), val);
  if (it == vec.end() || val < *it) {
    vec.insert(it, val);
    return true;
  }
  return false;
}

template <class T>
bool sorted_contains(std::vector<T> &vec, T val) {
    auto it = std::lower_bound(vec.begin(), vec.end(), val);
    return it != vec.end() && *it == val;
}

template <class T>
typename std::vector<T>::iterator sorted_find(std::vector<T> &vec, T val) {
    auto it = std::lower_bound(vec.begin(), vec.end(), val);
    if (it != vec.end() && val == *it) {
        return it;
    } else {
        return vec.end();
    }
}
