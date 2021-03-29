//
// Created by hryts on 22.03.21.
//

#ifndef IMPROVEDSTITCHING_KDTREE_H
#define IMPROVEDSTITCHING_KDTREE_H

// external includes
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <opencv2/opencv.hpp>

typedef uchar CoordinateType;
static const size_t g_dimensions = 128;

class KDPoint
{
public:
    KDPoint(CoordinateType* list, int descriptor_index) : m_descriptor_index(descriptor_index)
    {
        std::copy_n(list, g_dimensions, m_coordinates.begin());
    }

    [[nodiscard]] CoordinateType get_coordinate(size_t index) const
    {
        return m_coordinates[index];
    }

    [[nodiscard]] double distance(const KDPoint& pt) const
    {
        double dist = 0;
        for (size_t i = 0; i < g_dimensions; ++i)
            dist += pow(get_coordinate(i) - pt.get_coordinate(i), 2);
        return dist;
    }
    int get_descriptor_index() const
    {
    	return m_descriptor_index;
    }
private:
    int m_descriptor_index;
    std::array<CoordinateType, g_dimensions> m_coordinates{};
};

class KDTree
{
private:
    struct Node
    {
        explicit Node(const KDPoint& point) : m_point(point), m_left(nullptr), m_right(nullptr)
        {}

        [[nodiscard]] CoordinateType get_coordinate(size_t index) const
        {
            return m_point.get_coordinate(index);
        }

        [[nodiscard]] double distance(const KDPoint& pt) const
        {
            return m_point.distance(pt);
        }
        KDPoint m_point;
        Node* m_left;
        Node* m_right;
    };

    Node* m_root = nullptr;
    Node* m_best = nullptr;
    double m_best_dist = 0;
	std::vector<Node> m_nodes;

    struct Node_cmp
    {
        explicit Node_cmp(size_t index) : m_index(index) {}
        bool operator()(const Node& n1, const Node& n2) const
        {
            return n1.m_point.get_coordinate(m_index) < n2.m_point.get_coordinate(m_index);
        }
        size_t m_index;
    };

    Node* make_tree(size_t begin, size_t end, size_t index)
    {
        if (end <= begin)
            return nullptr;
        size_t n = begin + (end - begin)/2;
        std::nth_element(&m_nodes[begin], &m_nodes[n], &m_nodes[0] + end, Node_cmp(index));
        index = (index + 1) % g_dimensions;
        m_nodes[n].m_left = make_tree(begin, n, index);
        m_nodes[n].m_right = make_tree(n + 1, end, index);
        return &m_nodes[n];
    }

    void nearest(Node* root, const KDPoint& point, size_t index)
    {
        if (root == nullptr)
            return;
        double d = root->distance(point);
        if (m_best == nullptr || d < m_best_dist)
        {
            m_best_dist = d;
            m_best = root;
        }
        if (m_best_dist == 0)
            return;
        double dx = root->get_coordinate(index) - point.get_coordinate(index);
        index = (index + 1) % g_dimensions;
        nearest(dx > 0 ? root->m_left : root->m_right, point, index);
        if (dx * dx >= m_best_dist)
            return;
        nearest(dx > 0 ? root->m_right : root->m_left, point, index);
    }

public:
    KDTree(const KDTree&) = delete;
    KDTree& operator=(const KDTree&) = delete;

    explicit KDTree(const cv::Mat& image_descriptors)
    {
        for (int i = 0; i < image_descriptors.rows; ++i)
        {
            m_nodes.emplace_back(Node(KDPoint(image_descriptors.row(i).data, i)));
        }
        m_root = make_tree(0, m_nodes.size(), 0);
    }

    const KDPoint& nearest(const KDPoint& pt)
    {
        if (m_root == nullptr)
            throw std::logic_error("tree is empty");
        m_best = nullptr;
	    m_best_dist = 0;
        nearest(m_root, pt, 0);
        return m_best->m_point;
    }
};

#endif //IMPROVEDSTITCHING_KDTREE_H
