//
// Created by James Noeckel on 2/12/20.
//

#pragma once

#include <memory>
#include <algorithm>
#include <vector>
#include <limits>
#include <numeric>
#include <unordered_set>

template<typename T>
struct Interval {
    Interval(const T &start_p, const T &end_p) : start(start_p), end(end_p) {}

    T start;
    T end;
};

template<typename T, typename V>
struct IntervalComp {
    IntervalComp(const std::vector<std::pair<Interval<T>, V>> &intervals, const std::vector<size_t> &indices, bool sort_by_start) : intervals_(
            intervals), indices_(indices), sort_by_start_(sort_by_start) {}

    bool operator()(size_t a, size_t b) const {
        if (sort_by_start_) {
            return intervals_[indices_[a]].first.start < intervals_[indices_[b]].first.start;
        } else {
            return intervals_[indices_[a]].first.end < intervals_[indices_[b]].first.end;
        }
    }

    const std::vector<size_t> &indices_;
    const std::vector<std::pair<Interval<T>, V>> &intervals_;
    bool sort_by_start_;
};

/*template<typename T, typename V>
struct IntervalValueComp {
    IntervalValueComp(const std::vector<std::pair<Interval<T>, V>> &intervals, bool sort_by_start) : intervals_(
            intervals), sort_by_start_(sort_by_start) {}

    bool operator()(size_t a, size_t b) const {
        if (sort_by_start_) {
            return intervals_[a].first.start < intervals_[b].first.start;
        } else {
            return intervals_[a].first.end < intervals_[b].first.end;
        }
    }

    const std::vector<std::pair<Interval<T>, V> *> &intervals_;
    bool sort_by_start_;
};*/

/**
 * Query result structure: call begin() and end() to access iterator over result pairs.
 * Only valid as long as the interval tree is unchanged since calling query().
 */
template <typename T, typename V>
class IntervalTreeResult;

/**
 * An interval tree allows speedy intersection of a point on the number line with a collection of (possibly overlapping) intervals.
 * Intervals are inclusive on the left, exclusive on the right.
 * This structure acts as a tree multi-map with intersecting intervals as the keys.
 * @tparam T floating point type used by intervals
 * @tparam V stored value type
 */
template<typename T, typename V>
class IntervalTree {
    friend class IntervalTreeResult<T, V>;
public:
    IntervalTree() = default;

    ~IntervalTree() = default;

    IntervalTree(const IntervalTree &other);

    IntervalTree(IntervalTree &&other) noexcept;

    IntervalTree &operator=(const IntervalTree &other);

    IntervalTree &operator=(IntervalTree &&other) noexcept;

    template<typename ForwardIt>
    IntervalTree(ForwardIt begin, ForwardIt end);

    /**
     * Compute an interval tree with the given list of interval-value pairs.
     * @param intervals a collection of interval-value pairs where each `Interval(a, b)` represents the half-open interval [a, b).
     * Consequently, b must be greater than a.
     */
    template<typename ForwardIt>
    void build(ForwardIt begin, ForwardIt end);

    /**
     * Insert a single new value at the given interval. Note that this is a non-rebalancing tree, so if constructing a new tree
     * from a large number of elements, use build().
     * @param interval new interval (with positive length)
     * @param value value to store at the new interval
     */
    void insert(const Interval<T> &interval, const V &value);


    /**
     * Find all intervals intersecting with the query point. If the interval endpoints are a, b, return true if
     * a <= val < b
     * @param val query point
     * @return start and end forward iterators to solution pairs of intervals and values.
     */
    IntervalTreeResult<T, V> query(T val) const;

    /**
     * Find all intervals overlapping with query interval
     * @param interval
     */
    IntervalTreeResult<T, V> query(const Interval<T> &interval) const;

    size_t size() const;

    typename std::vector<std::pair<Interval<T>, V>>::const_iterator cbegin() const;
    typename std::vector<std::pair<Interval<T>, V>>::const_iterator cend() const;

    /**
     * Clears all data and resets the interval tree
     */
    void clear();

private:
    struct TreeNode;
    std::unique_ptr<TreeNode> root_;
    std::vector<std::pair<Interval<T>, V>> intervals_;
    std::vector<size_t> index_sorted_by_start_;
    std::vector<size_t> index_sorted_by_end_;
};

/* Definitions */

template<typename T, typename V>
IntervalTree<T, V>::IntervalTree(const IntervalTree &other) {
    root_ = other.root_->clone();
    intervals_ = other.intervals_;
    index_sorted_by_start_ = other.index_sorted_by_start_;
    index_sorted_by_end_ = other.index_sorted_by_end_;
}

template<typename T, typename V>
IntervalTree<T, V>::IntervalTree(IntervalTree &&other) noexcept {
    root_ = std::move(other.root_);
    intervals_ = std::move(other.intervals_);
    index_sorted_by_start_ = std::move(other.index_sorted_by_start_);
    index_sorted_by_end_ = std::move(other.index_sorted_by_end_);
}

template<typename T, typename V>
IntervalTree<T, V> &IntervalTree<T, V>::operator=(const IntervalTree &other) {
    if (this != &other) {
        root_ = other.root_->clone();
        intervals_ = other.intervals_;
        index_sorted_by_start_ = other.index_sorted_by_start_;
        index_sorted_by_end_ = other.index_sorted_by_end_;
    }
    return *this;
}

template<typename T, typename V>
IntervalTree<T, V> &IntervalTree<T, V>::operator=(IntervalTree &&other) noexcept {
    if (this != &other) {
        root_ = std::move(other.root_);
        intervals_ = std::move(other.intervals_);
        index_sorted_by_start_ = std::move(other.index_sorted_by_start_);
        index_sorted_by_end_ = std::move(other.index_sorted_by_end_);
    }
    return *this;
}

template<typename T, typename V>
template<typename ForwardIt>
IntervalTree<T, V>::IntervalTree(ForwardIt begin, ForwardIt end) {
    build(begin, end);
}

template<typename T, typename V>
template<typename ForwardIt>
void IntervalTree<T, V>::build(ForwardIt begin, ForwardIt end) {
    intervals_.clear();
    for (auto it = begin; it != end; it++) {
        //if (it->first.start < it->first.end) {
            intervals_.push_back(*it);
        //}
    }
    index_sorted_by_start_.resize(intervals_.size());
    index_sorted_by_end_.resize(intervals_.size());
    std::iota(index_sorted_by_start_.begin(), index_sorted_by_start_.end(), 0);
    std::iota(index_sorted_by_end_.begin(), index_sorted_by_end_.end(), 0);
    std::sort(index_sorted_by_start_.begin(), index_sorted_by_start_.end(), [&](size_t a, size_t b) {return intervals_[a].first.start < intervals_[b].first.start;});
    std::sort(index_sorted_by_end_.begin(), index_sorted_by_end_.end(), [&](size_t a, size_t b) {return intervals_[a].first.end < intervals_[b].first.end;});
    std::vector<size_t> indices(intervals_.size());
    std::iota(indices.begin(), indices.end(), 0);
    root_ = std::make_unique<TreeNode>(intervals_, indices);
}

template<typename T, typename V>
void IntervalTree<T, V>::insert(const Interval<T> &interval, const V &value) {
    //if (interval.start >= interval.end) return;
    intervals_.emplace_back(interval, value);
    index_sorted_by_start_.push_back(intervals_.size()-1);
    index_sorted_by_end_.push_back(intervals_.size()-1);
    std::sort(index_sorted_by_start_.begin(), index_sorted_by_start_.end(), [&](size_t a, size_t b) {return intervals_[a].first.start < intervals_[b].first.start;});
    std::sort(index_sorted_by_end_.begin(), index_sorted_by_end_.end(), [&](size_t a, size_t b) {return intervals_[a].first.end < intervals_[b].first.end;});
    if (root_) {
        root_->insert(intervals_, intervals_.size()-1);
    } else {
        root_ = std::make_unique<TreeNode>(intervals_, intervals_.size()-1);
    }
}

template<typename T, typename V>
IntervalTreeResult<T, V> IntervalTree<T, V>::query(T val) const {
    IntervalTreeResult<T, V> result;
    if (root_) {
        root_->query(intervals_, val, result);
    }
    return result;
}

template<typename T, typename V>
IntervalTreeResult<T, V> IntervalTree<T, V>::query(const Interval<T> &interval) const {
    IntervalTreeResult<T, V> result;
    //find intervals enclosing this interval
    if (root_) {
        //std::vector<bool> flag(intervals_.size(), false);
        std::unordered_set<const std::pair<Interval<T>, V> *> used;
        root_->query(intervals_, interval.start, result);
        for (auto ptr : result.results_) {
            used.insert(ptr);
        }
        //find the smallest left endpoint greater than or equal to query left endpoint
        {
            auto it = std::lower_bound(index_sorted_by_start_.begin(), index_sorted_by_start_.end(), interval.start,
                                       [&](size_t a, T val) { return intervals_[a].first.start < val; });
            for (; it != index_sorted_by_start_.end(); it++) {
                if (intervals_[*it].first.start < interval.end) {
                    if (used.find(&intervals_[*it]) == used.end()) {
                        result.results_.push_back(&intervals_[*it]);
                        used.insert(&intervals_[*it]);
                    }
                }
            }
        }
        {
            auto it = std::upper_bound(index_sorted_by_end_.begin(), index_sorted_by_end_.end(), interval.start,
                    [&] (T val, size_t b) {return val < intervals_[b].first.end;});
            for (; it != index_sorted_by_end_.end(); it++) {
                if (intervals_[*it].first.end < interval.end) {
                    if (used.find(&intervals_[*it]) == used.end()) {
                        result.results_.push_back(&intervals_[*it]);
                        used.insert(&intervals_[*it]);
                    }
                }
            }
        }
    }
    return result;
}

template<typename T, typename V>
size_t IntervalTree<T, V>::size() const {
    return intervals_.size();
}


template<typename T, typename V>
void IntervalTree<T, V>::clear() {
    root_.reset(nullptr);
    intervals_.clear();
    index_sorted_by_start_.clear();
    index_sorted_by_end_.clear();
}

//result type

template <typename T, typename V>
class IntervalTreeResult {
        friend class IntervalTree<T, V>::TreeNode;
        friend class IntervalTree<T, V>;
    public:
        using value_type = std::pair<Interval<T>, V>;
        struct Iterator {
            friend class IntervalTreeResult<T, V>;
            value_type operator*() {
                return *(parent_.results_[index_]);
            }
            const value_type *operator->() {
                return parent_.results_[index_];
            }
            Iterator &operator++() {
                index_++;
                return *this;
            }
            Iterator operator++(int) {
                Iterator copy = *this;
                ++*this;
                return copy;
            }
            bool operator==(const Iterator &other) {
                return index_ == other.index_;
            }
            bool operator!=(const Iterator &other) {
                return index_ != other.index_;
            }
        private:
            Iterator(size_t index, const IntervalTreeResult<T, V> &parent) : index_(index), parent_(parent) {}
            size_t index_;
            const IntervalTreeResult<T, V> &parent_;
        };
        /** start iterator of results */
        Iterator begin() const {
            return Iterator(0, *this);
        }
        /** past-the-end iterator of results */
        Iterator end() const {
            return Iterator(results_.size(), *this);
        }
        /** number of hits */
        size_t size() const {
            return results_.size();
        }
        void sort() {
            std::sort(results_.begin(), results_.end(), [](const value_type* a, const value_type* b) {return a->first.start < b->first.start;});
        }
    private:
        std::vector<const value_type*> results_;
};

//underlying structure

template<typename T, typename V>
struct IntervalTree<T, V>::TreeNode {
    void sort(const std::vector<std::pair<Interval<T>, V>> &intervals) {
        if (index_sorted_by_start_.size() < center_.size()) {
            size_t n = index_sorted_by_start_.size();
            index_sorted_by_start_.resize(center_.size());
            index_sorted_by_end_.resize(center_.size());
            std::iota(index_sorted_by_start_.begin() + n, index_sorted_by_start_.end(), n);
            std::iota(index_sorted_by_end_.begin() + n, index_sorted_by_end_.end(), n);
        }
        std::sort(index_sorted_by_start_.begin(), index_sorted_by_start_.end(), IntervalComp<T, V>(intervals, center_, true));
        std::sort(index_sorted_by_end_.begin(), index_sorted_by_end_.end(), IntervalComp<T, V>(intervals, center_, false));
    }

    TreeNode() = default;

    /**
     * Recursively construct an interval tree rooted at this node
     * @param intervals intervals to distribute among the nodes
     */
    explicit TreeNode(const std::vector<std::pair<Interval<T>, V>> &intervals, const std::vector<size_t> &indices) {
        T t_min = std::numeric_limits<T>::max();
        T t_max = std::numeric_limits<T>::lowest();
        for (auto index : indices) {
            t_min = std::min(t_min, intervals[index].first.start);
            t_max = std::max(t_max, intervals[index].first.end);
        }
        x_center_ = (t_min + t_max) / 2;
        if (x_center_ == t_max) x_center_ = t_min; //circumvents a numerical issue that leads to infinite depth
        std::vector<size_t> left;
        std::vector<size_t> right;
        for (auto i : indices) {
            if (intervals[i].first.end <= x_center_) {
                left.push_back(i);
            } else if (intervals[i].first.start > x_center_) {
                right.push_back(i);
            } else {
                center_.push_back(i);
            }
        }
        sort(intervals);
        if (!left.empty()) {
            left_ = std::make_unique<TreeNode>(intervals, left);
        }
        if (!right.empty()) {
            right_ = std::make_unique<TreeNode>(intervals, right);
        }
    }

    TreeNode(const std::vector<std::pair<Interval<T>, V>> &intervals, size_t index) {
        x_center_ = (intervals[index].first.start + intervals[index].first.end) / 2;
        if (x_center_ == intervals[index].first.end) x_center_ = intervals[index].first.start;
        center_.push_back(index);
        index_sorted_by_start_.push_back(0);
        index_sorted_by_end_.push_back(0);
    }

    void insert(const std::vector<std::pair<Interval<T>, V>> &intervals, size_t index) {
        if (intervals[index].first.end <= x_center_) {
            if (left_) {
                left_->insert(intervals, index);
            } else {
                left_ = std::make_unique<TreeNode>(intervals, index);
            }
        } else if (intervals[index].first.start > x_center_) {
            if (right_) {
                right_->insert(intervals, index);
            } else {
                right_ = std::make_unique<TreeNode>(intervals, index);
            }
        } else {
            center_.push_back(index);
            sort(intervals);
        }
    }

    void query(const std::vector<std::pair<Interval<T>, V>> &intervals, T val, IntervalTreeResult<T, V> &results) const {
        if (val <= x_center_) {
            for (auto it = index_sorted_by_start_.begin(); it != index_sorted_by_start_.end(); it++) {
                if (intervals[center_[*it]].first.start <= val) {
                    results.results_.push_back(&intervals[center_[*it]]);
                } else {
                    break;
                }
            }
            if (left_) {
                left_->query(intervals, val, results);
            }
        } else {
            for (auto it = index_sorted_by_end_.rbegin(); it != index_sorted_by_end_.rend(); it++) {
                if (intervals[center_[*it]].first.end > val) {
                    results.results_.push_back(&intervals[center_[*it]]);
                } else {
                    break;
                }
            }
            if (right_) {
                right_->query(intervals, val, results);
            }
        }
    }

    std::unique_ptr<TreeNode> clone() const {
        auto root = std::make_unique<TreeNode>();
        root->center_ = center_;
        root->index_sorted_by_start_ = index_sorted_by_start_;
        root->index_sorted_by_end_ = index_sorted_by_end_;
        root->x_center_ = x_center_;
        if (left_) {
            root->left_ = std::make_unique<TreeNode>();
            root->left_ = left_->clone();
        }
        if (right_) {
            root->right_ = std::make_unique<TreeNode>();
            root->right_ = right_->clone();
        }
        return root;
    }
    std::unique_ptr<TreeNode> left_, right_;
    std::vector<size_t> center_;
    std::vector<size_t> index_sorted_by_start_;
    std::vector<size_t> index_sorted_by_end_;
    T x_center_;
};

template<typename T, typename V>
typename std::vector<std::pair<Interval<T>, V>>::const_iterator
IntervalTree<T, V>::cbegin() const {
    return intervals_.cbegin();
}

template<typename T, typename V>
typename std::vector<std::pair<Interval<T>, V>>::const_iterator
IntervalTree<T, V>::cend() const {
    return intervals_.cend();
}
