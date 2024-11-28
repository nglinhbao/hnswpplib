#pragma once

#include "hnswlib.h"
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <numeric>
#include <future>
#include <omp.h>
// #include <chrono>

class HNSWPP {
public:
    HNSWPP(const int dim, 
           const size_t max_elements, 
           const int M = 16,
           const int ef_construction = 200,
           const int max_level = 4,
           const float scale_factor = 1.0f,
           const int ef_search = 20)
        : dim_(dim)
        , max_elements_(max_elements)
        , M_(M)
        , ef_construction_(ef_construction)
        , max_level_(std::floor((1 / std::log(M/2)) * std::log(max_elements/2)))
        , scale_factor_(1 / log(1.0 * M_))
        , space_(new hnswlib::L2Space(dim)) {
        
        // Initialize base layer and branches
        base_layer_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get(), max_elements_, M_, ef_construction_, false, std::max(1, static_cast<int>(std::round(ef_search / 2.0))));
        branch0_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get(), max_elements_, M_, ef_construction_, true, 1);
        branch1_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get(), max_elements_, M_, ef_construction_, true, 1);
    }

    // Prepare data and compute LID values
    void prepareIndex(const float* data) {
        // Compute pairwise distances
        std::vector<std::vector<float>> distances(max_elements_);
        for (size_t i = 0; i < max_elements_; i++) {
            distances[i].resize(max_elements_);
            for (size_t j = 0; j < max_elements_; j++) {
                float dist = 0.0f;
                for (int d = 0; d < dim_; d++) {
                    float diff = data[i * dim_ + d] - data[j * dim_ + d];
                    dist += diff * diff;
                }
                distances[i][j] = std::sqrt(dist);
            }
        }

        // Compute LID values and assign layers
        computeLIDValues(distances);

        // set normalized LID and average distance to branches
        branch0_->setNormalizedLID(normalized_lid_);
        branch0_->setAverageDistance(avg_distance_);
        branch1_->setNormalizedLID(normalized_lid_);
        branch1_->setAverageDistance(avg_distance_);
    }

    void addPoint(const float* point, hnswlib::labeltype label) {
        if (normalized_lid_.empty()) {
            throw std::runtime_error("LID values not computed. Call prepareIndex first.");
        }

        int layer = assigned_layers_[label];
        int branch = assigned_branches_[label];

        if (layer != 0) {
            if (branch == 0) {
                branch0_->setLevel(layer);
                branch0_->setConnectState(true);
                branch0_->addPoint(point, label);
                auto closest = branch0_->getClosestPoint();
                std::cout << "Closest point: " << closest << std::endl;
                base_layer_->setEnterpointNode(closest);
            } else {
                branch1_->setLevel(layer);
                branch1_->setConnectState(true);
                branch1_->addPoint(point, label);
                auto closest = branch1_->getClosestPoint();
                std::cout << "Closest point: " << closest << std::endl;
                base_layer_->setEnterpointNode(closest);
            }
        }
        else {
            if (branch == 0) {
                branch0_->setLevel(layer);
                branch0_->setConnectState(false);
                branch0_->addPoint(point, label);
                auto closest = branch0_->getClosestPoint();
                std::cout << "Closest point: " << closest << std::endl;
                base_layer_->setEnterpointNode(closest);
            } else {
                branch1_->setLevel(layer);
                branch1_->setConnectState(false);
                branch1_->addPoint(point, label);
                auto closest = branch1_->getClosestPoint();
                std::cout << "Closest point: " << closest << std::endl;
                base_layer_->setEnterpointNode(closest);
            }
        }
            
        base_layer_->addPoint(point, label);
    }

    std::priority_queue<std::pair<float, hnswlib::labeltype>> searchKnn(const float* query_data, const int k, const float lid_threshold) const {
        std::cout << "Query data: " << query_data << std::endl;
        // auto start_total = std::chrono::high_resolution_clock::now();

        // Set LID threshold for branches
        // auto start_setup = std::chrono::high_resolution_clock::now();
        branch0_->setLIDThreshold(lid_threshold);
        branch1_->setLIDThreshold(lid_threshold);
        // auto end_setup = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> setup_time = end_setup - start_setup;
        // std::cout << "Setup time: " << setup_time.count() << " ms" << std::endl;

        // Get results from both branches
        std::priority_queue<std::pair<float, hnswlib::labeltype>> branch0_results;
        std::priority_queue<std::pair<float, hnswlib::labeltype>> branch1_results;

        // auto start_branches = std::chrono::high_resolution_clock::now();
        
        // auto start_branch0 = std::chrono::high_resolution_clock::now();
        branch0_results = branch0_->searchKnn(query_data, 1, nullptr);
        // auto end_branch0 = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> branch0_time = end_branch0 - start_branch0;
        // std::cout << "Branch 0 search time: " << branch0_time.count() << " ms" << std::endl;

        // auto start_branch1 = std::chrono::high_resolution_clock::now();
        branch1_results = branch1_->searchKnn(query_data, 1, nullptr);
        // auto end_branch1 = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> branch1_time = end_branch1 - start_branch1;
        // std::cout << "Branch 1 search time: " << branch1_time.count() << " ms" << std::endl;

        // auto end_branches = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> total_branches_time = end_branches - start_branches;
        // std::cout << "Total parallel branches search time: " << total_branches_time.count() << " ms" << std::endl;
            
        // Store branch0 entry points
        // auto start_store_points = std::chrono::high_resolution_clock::now();
        std::vector<hnswlib::tableint> branch0_entry_points;
        while (!branch0_results.empty()) {
            branch0_entry_points.push_back(branch0_results.top().second);
            branch0_results.pop();
        }

        // Store branch1 entry points
        std::vector<hnswlib::tableint> branch1_entry_points;
        while (!branch1_results.empty()) {
            branch1_entry_points.push_back(branch1_results.top().second);
            branch1_results.pop();
        }
        // auto end_store_points = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> store_points_time = end_store_points - start_store_points;
        // std::cout << "Store entry points time: " << store_points_time.count() << " ms" << std::endl;

        // Create a priority queue for final results
        std::priority_queue<std::pair<float, hnswlib::labeltype>> final_results;

        // auto start_base = std::chrono::high_resolution_clock::now();

        // Using branch0 entry point
        std::unordered_set<hnswlib::labeltype> intermediate_exclude_set;

        if (!branch0_entry_points.empty()) {
            // auto start_base0_setup = std::chrono::high_resolution_clock::now();
            base_layer_->setEnterpointNode(branch0_entry_points[0]);
            // auto end_base0_setup = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double, std::milli> base0_setup_time = end_base0_setup - start_base0_setup;
            // std::cout << "Base layer 0 setup time: " << base0_setup_time.count() << " ms" << std::endl;

            // auto start_base0 = std::chrono::high_resolution_clock::now();
            auto results_from_branch0 = base_layer_->searchKnn(query_data, k);
            // auto end_base0 = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double, std::milli> base0_search_time = end_base0 - start_base0;
            // std::cout << "Base layer search from branch 0: " << base0_search_time.count() << " ms" << std::endl;
            
            // auto start_store0 = std::chrono::high_resolution_clock::now();
            // Store results and collect labels for exclude set
            while (!results_from_branch0.empty()) {
                auto result = results_from_branch0.top();
                final_results.push(result);
                intermediate_exclude_set.insert(result.second);  // Add to exclude set
                results_from_branch0.pop();
            }
            // auto end_store0 = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double, std::milli> store0_time = end_store0 - start_store0;
            // std::cout << "Store results from branch 0: " << store0_time.count() << " ms" << std::endl;
        }

        // Using branch1 entry point with updated exclude set
        if (!branch1_entry_points.empty()) {
            // auto start_base1_setup = std::chrono::high_resolution_clock::now();
            base_layer_->setEnterpointNode(branch1_entry_points[0]);
            base_layer_->setExcludeSet(intermediate_exclude_set);
            // auto end_base1_setup = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double, std::milli> base1_setup_time = end_base1_setup - start_base1_setup;
            // std::cout << "Base layer 1 setup time: " << base1_setup_time.count() << " ms" << std::endl;
            
            // auto start_base1 = std::chrono::high_resolution_clock::now();
            auto results_from_branch1 = base_layer_->searchKnn(query_data, k);
            // auto end_base1 = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double, std::milli> base1_search_time = end_base1 - start_base1;
            // std::cout << "Base layer search from branch 1: " << base1_search_time.count() << " ms" << std::endl;

            // auto start_store1 = std::chrono::high_resolution_clock::now();
            while (!results_from_branch1.empty()) {
                final_results.push(results_from_branch1.top());
                results_from_branch1.pop();
            }
            // auto end_store1 = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double, std::milli> store1_time = end_store1 - start_store1;
            // std::cout << "Store results from branch 1: " << store1_time.count() << " ms" << std::endl;
            
            // Clear exclude set after search
            base_layer_->setExcludeSet(std::unordered_set<hnswlib::labeltype>());
        }

        // auto end_base = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> total_base_time = end_base - start_base;
        // std::cout << "Total base layer search time: " << total_base_time.count() << " ms" << std::endl;

        // Combine and sort results
        // auto start_combine = std::chrono::high_resolution_clock::now();
        std::vector<std::pair<float, hnswlib::labeltype>> sorted_results;
        while (!final_results.empty()) {
            sorted_results.push_back(final_results.top());
            final_results.pop();
        }

        std::sort(sorted_results.begin(), sorted_results.end());
        auto last = std::unique(sorted_results.begin(), sorted_results.end(),
            [](const auto& a, const auto& b) { return a.second == b.second; });
        sorted_results.erase(last, sorted_results.end());

        // Create final priority queue with top k results
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result;
        for (int i = 0; i < std::min(k, (int)sorted_results.size()); i++) {
            result.push(sorted_results[i]);
        }
        // auto end_combine = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> combine_time = end_combine - start_combine;
        // std::cout << "Combine and sort results time: " << combine_time.count() << " ms" << std::endl;

        // auto end_total = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> total_time = end_total - start_total;
        // std::cout << "Total search time: " << total_time.count() << " ms" << std::endl;

        return result;
    }

    // Save index files
    void saveIndex(const std::string& filename_prefix) const {
        base_layer_->saveIndex(filename_prefix + "_base.bin");
        branch0_->saveIndex(filename_prefix + "_branch0.bin");
        branch1_->saveIndex(filename_prefix + "_branch1.bin");
    }

    // Load index files
    void loadIndex(const std::string& filename_prefix) {
        base_layer_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get(), filename_prefix + "_base.bin");
        branch0_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get(), filename_prefix + "_branch0.bin");
        branch1_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get(), filename_prefix + "_branch1.bin");
    }

private:
    // Helper function to compute LID
    static float computeLID(const std::vector<float>& distances) {
        if (distances.empty()) return 0.0f;
        
        std::vector<float> sorted_distances = distances;
        std::sort(sorted_distances.begin(), sorted_distances.end());
        
        const float epsilon = 1e-10;
        auto it = std::find_if(sorted_distances.begin(), sorted_distances.end(),
                             [epsilon](float d) { return d > epsilon; });
        if (it == sorted_distances.end()) return 0.0f;
        
        int k = std::min(20, static_cast<int>(sorted_distances.size() - 1));
        float r_k = sorted_distances[k];
        float sum_log = 0.0f;
        
        for (int i = 0; i < k; i++) {
            if (sorted_distances[i] > epsilon) {
                sum_log += std::log(sorted_distances[i] / r_k);
            }
        }
        
        return -k / sum_log;
    }

    // Helper function to normalize LID values
    static std::vector<float> normalizeLIDs(const std::vector<float>& lid_values) {
        if (lid_values.empty()) return {};
        
        float min_val = *std::min_element(lid_values.begin(), lid_values.end());
        float max_val = *std::max_element(lid_values.begin(), lid_values.end());
        
        if (max_val == min_val) {
            return std::vector<float>(lid_values.size(), 0.5f);
        }

        std::vector<float> normalized_lid(lid_values.size());
        for (size_t i = 0; i < lid_values.size(); i++) {
            normalized_lid[i] = (lid_values[i] - min_val) / (max_val - min_val);
        }
        return normalized_lid;
    }

    // Helper function to assign layers
    std::pair<std::vector<int>, std::vector<int>> assignLayers(
        const std::vector<float>& normalized_lid, 
        int max_level, 
        float scale_factor) const {
        
        size_t num_points = normalized_lid.size();
        std::vector<int> assigned_layers(num_points);
        std::vector<int> assigned_branches(num_points);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        std::vector<float> random_vals(num_points);
        for (size_t i = 0; i < num_points; i++) {
            random_vals[i] = dis(gen);
        }

        std::vector<int> expected_layer_size(max_level, 0);
        for (size_t i = 0; i < num_points; i++) {
            int layer = std::min(std::max(static_cast<int>(-std::log(random_vals[i]) * scale_factor), 0), 
                            max_level - 1);
            expected_layer_size[layer]++;
        }

        std::vector<size_t> sorted_indices(num_points);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(), 
                 [&normalized_lid](size_t i, size_t j) { return normalized_lid[i] > normalized_lid[j]; });

        std::vector<int> current_layer_size(max_level, 0);
        int assigned_branch = 0;

        for (size_t idx : sorted_indices) {
            int assigned_layer = 0;
            for (int layer = max_level - 1; layer >= 0; layer--) {
                if (current_layer_size[layer] < expected_layer_size[layer]) {
                    assigned_layer = layer;
                    break;
                }
            }
            
            assigned_layers[idx] = assigned_layer;
            assigned_branches[idx] = (assigned_branch % 2) ? 0 : 1;
            assigned_branch++;
            current_layer_size[assigned_layer]++;
        }

        printLayerStatistics(normalized_lid, assigned_layers, assigned_branches, max_level);
        return {assigned_layers, assigned_branches};
    }

    // Helper function to print layer statistics
    void printLayerStatistics(
        const std::vector<float>& normalized_lid,
        const std::vector<int>& assigned_layers,
        const std::vector<int>& assigned_branches,
        int max_level) const {
        
        std::cout << "Layer assignment completed: " << std::endl;
        for (int i = 0; i < max_level; i++) {
            float avg_lid = 0.0f;
            int count = 0;
            std::vector<int> branch_counts(2, 0);
            std::vector<float> branch_lids(2, 0.0f);
            
            for (size_t j = 0; j < normalized_lid.size(); j++) {
                if (assigned_layers[j] == i) {
                    avg_lid += normalized_lid[j];
                    count++;
                    int branch = assigned_branches[j];
                    branch_counts[branch]++;
                    branch_lids[branch] += normalized_lid[j];
                }
            }
            
            avg_lid = count > 0 ? avg_lid / count : 0.0f;
            float branch0_avg_lid = branch_counts[0] > 0 ? branch_lids[0] / branch_counts[0] : 0.0f;
            float branch1_avg_lid = branch_counts[1] > 0 ? branch_lids[1] / branch_counts[1] : 0.0f;
            
            std::cout << "Layer " << i << ": " << count << " nodes, avg LID: " << avg_lid << std::endl;
            std::cout << "  Branch 0: " << branch_counts[0] << " nodes, avg LID: " << branch0_avg_lid << std::endl;
            std::cout << "  Branch 1: " << branch_counts[1] << " nodes, avg LID: " << branch1_avg_lid << std::endl;
        }
    }

    void computeLIDValues(const std::vector<std::vector<float>>& distances) {
        size_t num_points = distances.size();
        lid_values_.resize(num_points);

        // Variables for computing avg_distance_
        float total_distance = 0.0f;
        size_t distance_count = 0;

        for (size_t i = 0; i < num_points; i++) {
            lid_values_[i] = computeLID(distances[i]);

            // Sum up all distances for avg_distance_
            for (float d : distances[i]) {
                total_distance += d;
                distance_count++;
            }
        }

        // Compute average distance
        avg_distance_ = distance_count > 0 ? total_distance / distance_count : 0.0f;

        // Normalize LID values and assign layers
        normalized_lid_ = normalizeLIDs(lid_values_);
        auto assignment = assignLayers(normalized_lid_, max_level_, scale_factor_);
        assigned_layers_ = std::move(std::get<0>(assignment));
        assigned_branches_ = std::move(std::get<1>(assignment));
    }

private:
    // Parameters
    const int dim_;
    const size_t max_elements_;
    const int M_;
    const int ef_construction_;
    const int max_level_;
    const float scale_factor_;

    // Space and indices
    std::unique_ptr<hnswlib::L2Space> space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> base_layer_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> branch0_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> branch1_;

    // LID and layer assignment
    std::vector<float> lid_values_;
    std::vector<float> normalized_lid_;
    std::vector<int> assigned_layers_;
    std::vector<int> assigned_branches_;
    float avg_distance_;
};