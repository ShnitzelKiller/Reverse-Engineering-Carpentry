//
// Created by James Noeckel on 11/19/19.
//


#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include "visualization.hpp"
#include "color_conversion.hpp"
#define CYLINDER_RES 16

enum {
    RAW_POINTS_INDEX = 0,
    CLUSTER_POINTS_INDEX = 1,
    BBOX_MESH_INDEX = 2,
    EDGES_INDEX = 3,
    SHAPES_INDEX = 4,
    MESH_INDEX = 5,
    SAMPLES_INDEX = 6
};

using namespace Eigen;

struct VisualizerImpl {
    VisualizerImpl() : colors_(0, 3) {}
    MatrixX3d colors_;
    igl::opengl::glfw::Viewer viewer_;
    igl::opengl::glfw::imgui::ImGuiMenu menu_;
    bool show_clusters_ = true;
    bool show_bboxes_ = true;
    bool show_points_ = true;
    bool show_shapes_ = true;
    bool show_edges_ = false;
    bool show_mesh_ = true;
    bool show_samples_ = true;
};

Visualizer::Visualizer(float point_size, int stride) : impl_(new VisualizerImpl) , stride_(stride) {
    impl_->viewer_.append_mesh(true);
    impl_->viewer_.append_mesh(true);
    impl_->viewer_.append_mesh(true);
    impl_->viewer_.append_mesh(true);
    impl_->viewer_.append_mesh(true);
    impl_->viewer_.data(RAW_POINTS_INDEX).point_size = point_size;
    impl_->viewer_.data(CLUSTER_POINTS_INDEX).point_size = point_size;
    impl_->viewer_.data(BBOX_MESH_INDEX);
    impl_->viewer_.data(EDGES_INDEX);
    impl_->viewer_.data(SHAPES_INDEX);
    impl_->viewer_.data(SAMPLES_INDEX).point_size = point_size * 3;
    impl_->viewer_.core().rotation_type = igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL;
    impl_->viewer_.selected_data_index = BBOX_MESH_INDEX;
    impl_->viewer_.plugins.push_back(&impl_->menu_);
    impl_->menu_.callback_draw_viewer_menu = [&]()
    {
        impl_->menu_.draw_viewer_menu();
        if (ImGui::CollapsingHeader("Show/Hide Groups", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Checkbox("Show clusters", &impl_->show_clusters_))
            {
                if (impl_->show_points_) {
                    impl_->viewer_.data(CLUSTER_POINTS_INDEX).set_visible(impl_->show_clusters_);
                    impl_->viewer_.data(RAW_POINTS_INDEX).set_visible(!impl_->show_clusters_);
                }
            }
            if (ImGui::Checkbox("Show bounding boxes", &impl_->show_bboxes_)) {
                impl_->viewer_.data(BBOX_MESH_INDEX).set_visible(impl_->show_bboxes_);
            }
            if (ImGui::Checkbox("Show points", &impl_->show_points_)) {
                if (impl_->show_points_) {
                    impl_->viewer_.data(CLUSTER_POINTS_INDEX).set_visible(impl_->show_clusters_);
                    impl_->viewer_.data(RAW_POINTS_INDEX).set_visible(!impl_->show_clusters_);
                } else {
                    impl_->viewer_.data(CLUSTER_POINTS_INDEX).set_visible(false);
                    impl_->viewer_.data(RAW_POINTS_INDEX).set_visible(false);
                }
            }
            if (ImGui::Checkbox("Show edges", &impl_->show_edges_)) {
                impl_->viewer_.data(EDGES_INDEX).set_visible(impl_->show_edges_);
            }
            if (ImGui::Checkbox("Show shapes", &impl_->show_shapes_)) {
                impl_->viewer_.data(SHAPES_INDEX).set_visible(impl_->show_shapes_);
            }
            if (ImGui::Checkbox("Show mesh", &impl_->show_mesh_)) {
                impl_->viewer_.data(MESH_INDEX).set_visible(impl_->show_mesh_);
            }
            if (ImGui::Checkbox("Show samples", &impl_->show_samples_)) {
                impl_->viewer_.data(SAMPLES_INDEX).set_visible(impl_->show_samples_);
            }
        }
    };
}

Visualizer::~Visualizer() = default;

void Visualizer::align_camera(const Eigen::Ref<const Eigen::MatrixX3d> &points) {
    impl_->viewer_.core().align_camera_center(points);
}

/*void Visualizer::visualize_points(pcl::const PointCloud&<pcl::PointXYZRGB>::ConstPtr cloud) {
    MatrixXd points(cloud->size()/stride_, 3);
    MatrixXd colors_loc(cloud->size()/stride_, 3);
    for (int i=0; i<cloud->size()/stride_; i+=1) {
        const auto &point = (*cloud)[i*stride_];
        points(i, 0) = point.x;
        points(i, 1) = point.y;
        points(i, 2) = point.z;
        colors_loc(i, 0) = point.r / 255.0;
        colors_loc(i, 1) = point.g / 255.0;
        colors_loc(i, 2) = point.b / 255.0;
    }
    impl_->viewer_.data(RAW_POINTS_INDEX).add_points(points, colors_loc);
}*/

void Visualizer::visualize_points(const PointCloud3& cloud) {
    Eigen::RowVector3d minPt = cloud.P.colwise().minCoeff();
    Eigen::RowVector3d maxPt = cloud.P.colwise().maxCoeff();
    MatrixXd points(cloud.P.rows()/stride_, 3);
    MatrixXd colors_n(cloud.P.rows()/stride_, 3);
    for (int i=0; i<cloud.P.rows()/stride_; i+=1) {
        points.row(i) = cloud.P.row(i);
        colors_n.row(i) = (cloud.P.row(i) - minPt).array()/(maxPt-minPt).array();
    }
    impl_->viewer_.data(RAW_POINTS_INDEX).add_points(points, colors_n);
}

void Visualizer::visualize_clusters(const PointCloud3& cloud, const std::vector<std::vector<int>> &clusters) {
    update_colors(clusters.size());
    for (size_t c=0; c < clusters.size(); c++) {
        MatrixXd points(clusters[c].size()/stride_, 3);
        for (size_t i=0; i<clusters[c].size()/stride_; i++) {
            points.row(i) = cloud.P.row(clusters[c][i * stride_]).head(3);
        }
        impl_->viewer_.data(CLUSTER_POINTS_INDEX).add_points(points, impl_->colors_.row(c));
    }
}

void Visualizer::visualize_primitives(const std::vector<BoundedPlane> &bboxes, const std::vector<Cylinder> &cylinders) {
    update_colors(bboxes.size() + cylinders.size());
    std::vector<Vector3d> verts;
    std::vector<Vector3i> faces;
    std::vector<Vector3d> colors;
    MatrixXd P1(bboxes.size(), 3);
    MatrixXd P2(bboxes.size(), 3);
    size_t voffset = 0;
    for (size_t c=0; c<bboxes.size(); c++) {
        const auto &bbox = bboxes[c];
        if (bbox.hasShape()) {
            MatrixX3d pts = bbox.points3D().cast<double>();
            for (size_t i = 0; i < pts.rows(); i++) {
                verts.emplace_back(pts.row(i).transpose());
            }
            for (size_t i = 1; i < pts.rows() - 1; i++) {
                faces.emplace_back(voffset, voffset + i + 1, voffset + i);
                colors.emplace_back(impl_->colors_.row(c));
            }
            voffset += pts.rows();

            P1.row(c) = 0.5 * (pts.row(0) + pts.row(pts.rows() / 2));
            P2.row(c) = P1.row(c) +
                        0.25 * (pts.row(0) - pts.row(pts.rows() / 2)).norm() * bbox.basis().row(2).cast<double>();
        }
    }
    for (size_t c=0; c<cylinders.size(); c++) {
        Vector3d up(0, 0, 0);
        int ind = 0;
        float mincomp = std::abs(cylinders[c].dir()[0]);
        for (int j = 1; j < 3; j++) {
            float abscomp = std::abs(cylinders[c].dir()[j]);
            if (abscomp < mincomp) {
                ind = j;
                mincomp = abscomp;
            }
        }
        up[ind] = 1.0f;
        Vector3d u = cylinders[c].dir().cast<double>().cross(up).normalized();
        Vector3d v = u.cross(cylinders[c].dir().cast<double>());
        for (size_t i=0; i<CYLINDER_RES; i++) {
            double ang = static_cast<double>(i)/CYLINDER_RES * 2 * M_PI;
            double x = cos(ang) * cylinders[c].radius();
            double y = sin(ang) * cylinders[c].radius();
            verts.emplace_back(cylinders[c].point().cast<double>() + x * u + y * v + cylinders[c].dir().cast<double>() * cylinders[c].start());
            verts.emplace_back(cylinders[c].point().cast<double>() + x * u + y * v + cylinders[c].dir().cast<double>() * cylinders[c].end());
            faces.emplace_back(
                    Vector3i(voffset + i*2,
                                    voffset + i*2+1,
                                    voffset + ((i+1)%CYLINDER_RES)*2));
            faces.emplace_back(
                    Vector3i(voffset + i*2+1,
                                    voffset + ((i+1)%CYLINDER_RES)*2+1,
                                    voffset + ((i+1)%CYLINDER_RES)*2));
            colors.emplace_back(impl_->colors_.row(bboxes.size() + c));
            colors.emplace_back(impl_->colors_.row(bboxes.size() + c));
        }
        voffset += 2*CYLINDER_RES;
    }

    MatrixX3d V(verts.size(), 3);
    MatrixX3i F(faces.size(), 3);
    MatrixX3d C(faces.size(), 3);
    for (size_t i=0; i<verts.size(); i++) {
        V.row(i) = verts[i].transpose();
    }
    for (size_t i=0; i<faces.size(); i++) {
        F.row(i) = faces[i].transpose();
        C.row(i) = colors[i].transpose();
    }
    impl_->viewer_.data(BBOX_MESH_INDEX).set_mesh(V, F);
    impl_->viewer_.data(BBOX_MESH_INDEX).set_colors(C);
    impl_->viewer_.data(BBOX_MESH_INDEX).add_edges(P1, P2, RowVector3d(0., 0., 1.));
}

void Visualizer::visualize_shapes(const std::vector<std::vector<std::vector<Vector3d>>> &allcontours) {
    update_colors(allcontours.size());
    for (size_t c=0; c<allcontours.size(); c++) {
        const auto &contours = allcontours[c];
        size_t total_points = 0;
        for (const auto &contour : contours) {
            total_points += contour.size();
        }
        MatrixX3d P1(total_points, 3);
        MatrixX3d P2(total_points, 3);
        size_t offset = 0;
        for (const auto &contour : contours) {
            for (size_t i = 0; i < contour.size(); i++) {
                P1.row(offset + i) = contour[i].transpose();
                P2.row(offset + i) = contour[(i + 1) % contour.size()].transpose();
            }
            offset += contour.size();
        }
        impl_->viewer_.data(SHAPES_INDEX).add_edges(P1, P2, impl_->colors_.row(c));
    }
}

void Visualizer::visualize_shapes(const std::vector<std::vector<Edge3d>> &allcontours) {
    update_colors(allcontours.size());
    for (size_t c=0; c<allcontours.size(); c++) {
        const auto &contour = allcontours[c];
        size_t total_points = contour.size();
        MatrixX3d P1(total_points, 3);
        MatrixX3d P2(total_points, 3);
        for (size_t i=0; i < contour.size(); i++) {
            const auto &edge = contour[i];
            P1.row(i) = edge.first.transpose();
            P2.row(i) = edge.second.transpose();
        }
        impl_->viewer_.data(SHAPES_INDEX).add_edges(P1, P2, impl_->colors_.row(c));
    }
}



void Visualizer::visualize_edges(const std::vector<Edge3d> &edges, const std::vector<Vector3d> &color) {
    MatrixXd P1(edges.size(), 3);
    MatrixXd P2(edges.size(), 3);
    for (size_t i=0; i<edges.size(); i++) {
        P1.row(i) = edges[i].first.transpose();
        P2.row(i) = edges[i].second.transpose();
    }
    MatrixX3d color_mat(color.size(), 3);
    for (size_t i=0; i<color.size(); i++) {
        color_mat.row(i) = color[i].transpose();
    }
    impl_->viewer_.data(EDGES_INDEX).add_edges(P1, P2, color_mat);
}

void Visualizer::update_colors(size_t size) {
    if (impl_->colors_.rows() < size) {
        size_t oldSize = impl_->colors_.rows();
        impl_->colors_.resize(size, 3);
        for (size_t i = oldSize; i < size; i++) {
            const unsigned char h = rand() % 360;
            float intensity = static_cast<float>((rand() % 128) + 128) / 255.0f;
            float rf, gf, bf;
            hsv2rgb<float>(static_cast<float>(h), 1.0f, 1.0f, rf, gf, bf);
            float r = (rf * intensity);
            float g = (gf * intensity);
            float b = (bf * intensity);
            impl_->colors_.row(i) = RowVector3d(r, g, b);
        }
    }
}

Vector3d Visualizer::color(size_t i) const {
    return impl_->colors_.row(i).transpose();
}

MatrixX3d Visualizer::colors() const {
    return impl_->colors_;
}

void Visualizer::launch() {
    impl_->viewer_.data(CLUSTER_POINTS_INDEX).set_visible(impl_->show_clusters_ && impl_->show_points_);
    impl_->viewer_.data(RAW_POINTS_INDEX).set_visible(!impl_->show_clusters_ && impl_->show_points_);
    impl_->viewer_.data(BBOX_MESH_INDEX).set_visible(impl_->show_bboxes_);
    impl_->viewer_.data(SHAPES_INDEX).set_visible(impl_->show_shapes_);
    impl_->viewer_.data(EDGES_INDEX).set_visible(impl_->show_edges_);
    impl_->viewer_.data(MESH_INDEX).set_visible(impl_->show_mesh_);
    impl_->viewer_.data(SAMPLES_INDEX).set_visible(impl_->show_samples_);
    impl_->viewer_.launch();
}

void Visualizer::visualize_sample_points(const PointCloud3& cloud, const Ref<const MatrixX3d> &colors) {
    impl_->viewer_.data(SAMPLES_INDEX).add_points(cloud.P.block(0, 0, cloud.P.rows(), 3), colors);
}

void Visualizer::visualize_sample_points(const PointCloud3& cloud, std::vector<size_t> &indices,
                                         const Ref<const MatrixX3d> &colors) {
    MatrixX3d points(indices.size(), 3);
    for (size_t i=0; i<indices.size(); i++) {
        points.row(i) = cloud.P.row(indices[i]).head(3);
    }
    impl_->viewer_.data(SAMPLES_INDEX).add_points(points, colors);
}

void Visualizer::visualize_mesh(const Ref<const MatrixX3d> &V, const Ref<const MatrixX3i> &F) {
    impl_->viewer_.data(MESH_INDEX).set_mesh(V, F);
}

void Visualizer::clear_mesh() {
    impl_->viewer_.data(BBOX_MESH_INDEX).clear();
}
