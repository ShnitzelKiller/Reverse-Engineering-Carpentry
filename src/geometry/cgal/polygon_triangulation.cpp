//
// Created by James Noeckel on 8/19/20.
//

#include "geom.h"
#include <CGAL/Constrained_triangulation_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>

struct FaceInfo2
{
    FaceInfo2(){}
    int nesting_level;
    bool in_domain(){
        return nesting_level % 2 == 1;
    }
};

template < class Gt, class Vb = CGAL::Triangulation_vertex_base_2<Gt> >
class Vertex : public  Vb {
    typedef Vb superclass;
public:
    typedef typename Vb::Vertex_handle      Vertex_handle;
    typedef typename Vb::Point              Point;

    template < typename TDS2 >
    struct Rebind_TDS {
        typedef typename Vb::template Rebind_TDS<TDS2>::Other Vb2;
        typedef Vertex<Gt, Vb2> Other;
    };

public:
    Vertex() : superclass() {}
    Vertex(const Point & p) : superclass(p) {}
    int index;
};
typedef Vertex<Kernel>                                                  Vb;
typedef CGAL::Triangulation_face_base_with_info_2<FaceInfo2, Kernel>    Fbb;
typedef CGAL::Constrained_triangulation_face_base_2<Kernel, Fbb>        Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb>                    TDS;
typedef CGAL::Exact_predicates_tag                                      Itag;
typedef CGAL::Constrained_triangulation_2<Kernel, TDS, Itag>            CT;


void insert_polygon(CT& ct, const Polygon_2& polygon, std::vector<int> &indexInt){
    if (polygon.is_empty()) return;
    int index = 0;

    CT::Vertex_handle v_prev = ct.insert(*CGAL::cpp11::prev(polygon.vertices_end()));

    for (Polygon_2::Vertex_iterator vit = polygon.vertices_begin();
         vit != polygon.vertices_end(); ++vit)
    {
        CT::Vertex_handle vh = ct.insert(*vit);
        vh->index = indexInt[index];
        index++;
        ct.insert_constraint(vh, v_prev);
        v_prev = vh;
    }
}

void
mark_domains(CT& ct,
             CT::Face_handle start,
             int index,
             std::list<CT::Edge>& border)
{
    if (start->info().nesting_level != -1){
        return;
    }
    std::list<CT::Face_handle> queue;
    queue.push_back(start);
    while (!queue.empty()){
        CT::Face_handle fh = queue.front();
        queue.pop_front();
        if (fh->info().nesting_level == -1){
            fh->info().nesting_level = index;
            for (int i = 0; i < 3; i++){
                CT::Edge e(fh, i);
                CT::Face_handle n = fh->neighbor(i);
                if (n->info().nesting_level == -1){
                    if (ct.is_constrained(e)) border.push_back(e);
                    else queue.push_back(n);
                }
            }
        }
    }
}
//explore set of facets connected with non constrained edges,
//and attribute to each such set a nesting level.
//We start from facets incident to the infinite vertex, with a nesting
//level of 0. Then we recursively consider the non-explored facets incident
//to constrained edges bounding the former set and increase the nesting level by 1.
//Facets in the domain are those with an odd nesting level.
void
mark_domains(CT& ct)
{
    for (CT::All_faces_iterator it = ct.all_faces_begin(); it != ct.all_faces_end(); ++it){
        it->info().nesting_level = -1;
    }
    std::list<CT::Edge> border;
    mark_domains(ct, ct.infinite_face(), 0, border);
    while (!border.empty()){
        CT::Edge e = border.front();
        border.pop_front();
        CT::Face_handle n = e.first->neighbor(e.second);
        if (n->info().nesting_level == -1){
            mark_domains(ct, n, e.first->info().nesting_level + 1, border);
        }
    }
}

void polygon_triangulation(const std::vector<std::vector<Eigen::Vector2d>> &polys, std::vector<Eigen::Vector3i> &facets) {
    CT ct;
    int nb = 0;
    for (int i = 0; i < polys.size(); i++)
    {
        Polygon_2 polygon;
        std::vector<int> indexInt;
        for (int j = 0; j < polys[i].size(); j++)
        {
            polygon.push_back(Point_2(polys[i][j][0], polys[i][j][1]));
            indexInt.emplace_back(j + nb);
        }
        nb += polys[i].size();
        insert_polygon(ct, polygon, indexInt);
    }

    //Mark facets that are inside the domain bounded by the polygon
    mark_domains(ct);

    for (CT::Finite_faces_iterator fit = ct.finite_faces_begin();
         fit != ct.finite_faces_end(); ++fit)
        if (fit->info().in_domain())
            facets.emplace_back(Eigen::Vector3i(fit->vertex(2)->index, fit->vertex(1)->index, fit->vertex(0)->index));
}