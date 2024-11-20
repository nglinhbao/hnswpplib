#include "../../src/hnswlib.h"
#include <vector>
#include <algorithm>
#include <iostream>

int main() {
    try {
        std::cout << "Starting program..." << std::endl;
        
        int dim = 16;               // Dimension of the elements
        int max_elements = 10000;   // Maximum number of elements
        int M = 16;                 // Graph degree
        int ef_construction = 200;  // Construction time/accuracy trade-off
        int max_level = 4;         // Maximum level for layer assignment
        float scale_factor = 1.0f;  // Scale factor for layer distribution
        int k_neighbors = 100;      // Number of nearest neighbors for LID calculation

        std::cout << "Initializing space and index..." << std::endl;
        
        // Initialize index
        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

        std::cout << "Generating random data..." << std::endl;
        
        // Generate random data
        std::mt19937 rng(47);
        std::uniform_real_distribution<> distrib_real;
        float* data = new float[dim * max_elements];
        if (!data) {
            throw std::runtime_error("Failed to allocate data array");
        }
        
        for (int i = 0; i < dim * max_elements; i++) {
            data[i] = distrib_real(rng);
        }

        std::cout << "Calculating distances..." << std::endl;
        
        // Calculate distances for each point to every other point
        std::vector<std::vector<float>> distances(max_elements);
        std::vector<hnswlib::labeltype> point_labels(max_elements);
        
        // Initialize distance vectors and labels
        for (int i = 0; i < max_elements; i++) {
            distances[i].resize(k_neighbors);
            point_labels[i] = i;
        }

        // Calculate k nearest neighbor distances for each point
        for (int i = 0; i < max_elements; i++) {
            if (i % 1000 == 0) {
                std::cout << "Processing point " << i << "/" << max_elements << std::endl;
            }
            
            std::vector<std::pair<float, int>> point_distances;
            point_distances.reserve(max_elements - 1);  // Reserve space for all points except self
            
            // Calculate distances to all other points
            const float* current_point = data + i * dim;
            for (int j = 0; j < max_elements; j++) {
                if (i != j) {
                    const float* other_point = data + j * dim;
                    float dist = 0.0f;
                    
                    // Manual distance calculation to avoid SIMD issues
                    for (int d = 0; d < dim; d++) {
                        float diff = current_point[d] - other_point[d];
                        dist += diff * diff;
                    }
                    point_distances.emplace_back(dist, j);
                }
            }
            
            // Sort distances and keep k nearest
            if (k_neighbors > point_distances.size()) {
                throw std::runtime_error("k_neighbors larger than available points");
            }
            
            std::partial_sort(point_distances.begin(), 
                            point_distances.begin() + k_neighbors, 
                            point_distances.end(),
                            [](const auto& a, const auto& b) { return a.first < b.first; });
            
            // Store k nearest distances
            for (int k = 0; k < k_neighbors; k++) {
                distances[i][k] = point_distances[k].first;
            }
        }

        std::cout << "Computing and assigning layers..." << std::endl;
        
        // Compute and assign layers
        alg_hnsw->computeAndAssignLayers(distances, point_labels, max_level, scale_factor);

        std::cout << "Adding points to index..." << std::endl;
        
        // Add data to index
        for (int i = 0; i < max_elements; i++) {
            alg_hnsw->addPoint(data + i * dim, i, 0);
            if (i % 1000 == 0) {
                std::cout << "Added point " << i << "/" << max_elements << std::endl;
            }
        }

        std::cout << "Testing recall..." << std::endl;
        
        // Query the elements for themselves and measure recall
        float correct = 0;
        for (int i = 0; i < max_elements; i++) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = 
                alg_hnsw->searchKnn(data + i * dim, 1);
            hnswlib::labeltype label = result.top().second;
            if (label == i) correct++;
            
            if (i % 1000 == 0) {
                std::cout << "Tested point " << i << "/" << max_elements << std::endl;
            }
        }
        
        float recall = correct / max_elements;
        std::cout << "Recall: " << recall << "\n";

        // std::cout << "Saving index..." << std::endl;
        
        // // Serialize index
        // std::string hnsw_path = "hnsw.bin";
        // alg_hnsw->saveIndex(hnsw_path);
        // delete alg_hnsw;

        // std::cout << "Loading saved index..." << std::endl;
        
        // // Deserialize index and check recall
        // alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
        // correct = 0;
        // for (int i = 0; i < max_elements; i++) {
        //     std::priority_queue<std::pair<float, hnswlib::labeltype>> result = 
        //         alg_hnsw->searchKnn(data + i * dim, 1);
        //     hnswlib::labeltype label = result.top().second;
        //     if (label == i) correct++;
            
        //     if (i % 1000 == 0) {
        //         std::cout << "Tested loaded point " << i << "/" << max_elements << std::endl;
        //     }
        // }
        
        // recall = (float)correct / max_elements;
        // std::cout << "Recall of deserialized index: " << recall << "\n";

        // delete[] data;
        // delete alg_hnsw;
        
        // std::cout << "Program completed successfully" << std::endl;
        // return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}