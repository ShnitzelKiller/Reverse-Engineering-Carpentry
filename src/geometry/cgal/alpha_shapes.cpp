#include "geom.h"
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Alpha_shape_vertex_base_2.h>
#include <CGAL/Alpha_shape_face_base_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/algorithm.h>
#include <CGAL/assertions.h>
#include <vector>
/*#include <CGAL/boost/iterator/counting_iterator.hpp>
#include <CGAL/Search_traits_2.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/point_generators_2.h>
#include <CGAL/Kd_tree.h>

#include <CGAL/Fuzzy_sphere.h>*/

#include <CGAL/Quotient.h>
#include <CGAL/Arr_segment_traits_2.h>
#include <CGAL/Arrangement_2.h>

typedef CGAL::Alpha_shape_vertex_base_2<Kernel> Vb;
typedef CGAL::Alpha_shape_face_base_2<Kernel> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Delaunay_triangulation_2<Kernel, Tds> Triangulation_2;
typedef CGAL::Alpha_shape_2<Triangulation_2> Alpha_shape_2;
typedef Alpha_shape_2::Alpha_shape_edges_iterator Alpha_shape_edges_iterator;

/*class My_point_property_map {
    const std::vector<Segment> &segments;
public:
    typedef Point value_type;
    typedef const value_type &reference;
    typedef std::size_t key_type;
    typedef boost::lvalue_property_map_tag category;

    explicit My_point_property_map(const std::vector<Segment> &segs) : segments(segs) {}

    reference operator[](key_type k) const { return segments[k].source(); }

    friend reference get(const My_point_property_map &ppmap, key_type i) { return ppmap[i]; }
};

typedef CGAL::Search_traits_2<Kernel> Traits_base;
typedef CGAL::Search_traits_adapter<std::size_t, My_point_property_map, Traits_base> Traits;
typedef CGAL::Sliding_midpoint<Traits> Splitter;
typedef CGAL::Kd_tree<Traits, Splitter, CGAL::Tag_false> Kd_Tree;
typedef CGAL::Fuzzy_sphere<Traits> Fuzzy_sphere;*/

/*typedef CGAL::Arr_segment_traits_2<Kernel> Traits_2;
typedef Traits_2::Point_2 Point;
typedef Traits_2::X_monotone_curve_2 Segment;
typedef CGAL::Arrangement_2<Traits_2> Arrangement_2;*/

template<class OutputIterator>
void alpha_edges(const Alpha_shape_2 &A, OutputIterator out) {
    Alpha_shape_edges_iterator it = A.alpha_shape_edges_begin(),
            end = A.alpha_shape_edges_end();
    for (; it != end; ++it)
        *out++ = A.segment(*it);
}

void
alpha_shapes(const Eigen::Ref<const Eigen::MatrixX2d> &shape, std::vector<std::vector<Eigen::Vector2d>> &edgepoints, std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> &raw_edges, float alpha) {
    std::vector<Point_2> points;
    points.reserve(shape.rows());
    for (size_t i = 0; i < shape.rows(); i++) {
        Point_2 point(shape(i, 0), shape(i, 1));
        points.push_back(point);
    }
    Alpha_shape_2 A;
    A.set_mode(Alpha_shape_2::REGULARIZED);
    A.make_alpha_shape(points.begin(), points.end());
    if (alpha < 0) {
        //DEBUG:
        std::cout << "number of alphas: " << A.number_of_alphas() << std::endl;
        auto it = A.find_optimal_alpha(1);
        if (it == A.alpha_end()) {
            std::cerr << "no alpha found" << std::endl;
            return;
        } else {
            std::cout << "optimal alpha: " << *it << std::endl;
        }
        ///
        A.set_alpha(*it);
    } else {
        A.set_alpha(alpha);
    }
    std::vector<Segment> segments;
    alpha_edges(A, std::back_inserter(segments));
    std::cout << "total segments: " << segments.size() << std::endl;
    for (const auto &seg : segments) {
        raw_edges.emplace_back(Eigen::Vector2d(seg.source().x(), seg.source().y()), Eigen::Vector2d(seg.target().x(), seg.target().y()));
    }
    //ATTEMPT TO USE DCELs
    /*std::cout << "constructing DCEL from " << segments.size() << " edges" << std::endl;
    Arrangement_2 arr;
    CGAL::insert(arr, segments.begin(), segments.end());
    std::cout << "number of faces: " << arr.number_of_faces() << std::endl;
    for (auto faces_it = arr.faces_begin(); faces_it != arr.faces_end(); faces_it++) {
        size_t num_ccbs = faces_it->number_of_outer_ccbs();
        if (num_ccbs == 1) {
            auto circulator = faces_it->outer_ccb();
            auto curr = circulator;
            size_t num_he = 0;
            do {
                num_he++;
            } while (++curr != circulator);
            std::cout << "size " << num_he << std::endl;
        } else {
            std::cout << "face has " << num_ccbs << " ccbs" << std::endl;
        }
    }*/
    //ATTEMPT TO GLUE WITH MAP
    //TODO: Hierarchically separate curves; for now just concatenate them
    std::map<Point_2, std::vector<size_t>> seg_map;
    for (size_t i=0; i<segments.size(); i++) {
        seg_map[segments[i].source()].push_back(i);
        seg_map[segments[i].target()].push_back(i);
    }
    size_t warning_pts = 0;
    for (auto &pair : seg_map) {
        if (pair.second.size() < 2) {
            warning_pts++;
        }
    }
    if (warning_pts > 0) {
        std::cerr << "WARNING: " << warning_pts << " points have less than 2 neighbors" << std::endl;
    }
    size_t curr_seg = 0;
    size_t curr_loop_base = 0;
    size_t total_segments = 0;
    std::unordered_set<size_t> visited;
    bool ending_on_target = true;
    while (total_segments < segments.size()) {
        edgepoints.emplace_back();
        for (; total_segments < segments.size();) {
            auto &segment = segments[curr_seg];
            visited.insert(curr_seg);
            //std::cout << "segment " << segment.source().x() << ", " << segment.source().y() << " - " << segment.target().x() << ", " << segment.target().y() << std::endl;
            edgepoints[edgepoints.size()-1].emplace_back(segment.source().x(), segment.source().y());
            std::map<Point_2, std::vector<size_t>>::iterator it;
            if (ending_on_target) {
                it = seg_map.find(segment.target());
            } else {
                it = seg_map.find(segment.source());
            }
            if (it == seg_map.end()) {
                std::cerr << "failed to finish curve" << std::endl;
                break;
            }
            if (it->second[0] == curr_seg) {
                if (it->second.size() > 1) {
                    curr_seg = it->second[1];
                } else {
                    std::cerr << "no neighboring segment; aborting" << std::endl;
                    break;
                }
            } else {
                curr_seg = it->second[0];
            }
            Point_2 endpoint = ending_on_target ? segment.target() : segment.source();
            ending_on_target = endpoint == segments[curr_seg].source();
            total_segments++;
            if (curr_seg == curr_loop_base) {
                break;
            }
        }
        bool found = false;
        for (size_t i=0; i<segments.size(); i++) {
            if (visited.find(i) == visited.end()) {
                curr_seg = i;
                curr_loop_base = i;
                found = true;
                break;
            }
        }
        if (!found) {
            break;
        }
    }


    //ATTEMPT TO GLUE WITH KD-TREE
    /*My_point_property_map ppmap(segments);
    std::cout << "building tree with " << segments.size() << " elements...";
    Kd_Tree tree(
            boost::counting_iterator<std::size_t>(0),
            boost::counting_iterator<std::size_t>(points.size()),
            Splitter(),
            Traits(ppmap));
    std::cout << "done" << std::endl;
    size_t curr_segment = 0;
    for (size_t i=0; i<segments.size(); i++) {
        pcl::PointXY pt;
        pt.x = segments[curr_segment].source().x();
        pt.y = segments[curr_segment].source().y();
        curve->push_back(pt);
        Point_2 nextsrc = segments[curr_segment].target();
        std::cout << "searching for " << nextsrc.x() << ", " << nextsrc.y() << std::endl;
        auto opt = tree.search_any_point(Fuzzy_sphere(nextsrc, 0, 0, Traits(ppmap)));
        std::cout << "found" << std::endl;
        if (opt.has_value()) {
            curr_segment = opt.get();
        } else {
            std::cerr << "failed to find all segments" << std::endl;
            return;
        }
        std::cout << "processed " << i << std::endl;
    }*/
}
