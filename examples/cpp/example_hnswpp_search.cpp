#include "../../src/hnswppalg.h"
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <cmath>
#include <numeric>
#include <iomanip>

int main() {
    int dim = 16;               // Dimension of the elements
    int max_elements = 1000;    // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                               // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int max_level = 10;         // Maximum number of layers
    float scale_factor = 1.0f; // Scale factor for layer assignment
    int ef_search = 20;        
    float lid_threshold = 0.5f; // Threshold for LID values

    // Initing index
    HNSWPP* index = new HNSWPP(dim, max_elements, M, ef_construction, max_level, scale_factor, ef_search);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    // Prepare index (computes distances and LID values internally)
    std::cout << "Preparing index (computing distances and LID values)..." << std::endl;
    index->prepareIndex(data);

    // Add data to index
    std::cout << "Adding points to index..." << std::endl;
    for (int i = 0; i < max_elements; i++) {
        index->addPoint(data + i * dim, i);
    }

    int k = 10;
    // Query the elements for themselves and measure recall
    std::cout << "Measuring recall..." << std::endl;
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        auto result = index->searchKnn(data + i * dim, k, lid_threshold);
        
        // Check if point i appears in any of the k results
        bool found = false;
        std::vector<std::pair<float, hnswlib::labeltype>> temp_results;
        while (!result.empty()) {
            if (result.top().second == i) {
                found = true;
                break;
            }
            temp_results.push_back(result.top());
            result.pop();
        }
        if (found) correct++;
    }

    float recall = correct / max_elements;
    std::cout << "k value: " << k << std::endl;
    std::cout << "Recall@" << k << ": " << recall << "\n";

    // Serialize index
    std::cout << "\nSaving index..." << std::endl;
    std::string hnsw_path = "hnswpp";
    index->saveIndex(hnsw_path);
    delete index;

    // Deserialize index and check recall
    std::cout << "Loading index..." << std::endl;
    index = new HNSWPP(dim, max_elements, M, ef_construction, max_level, scale_factor);
    index->loadIndex(hnsw_path);
    
    correct = 0;
    std::cout << "Testing loaded index..." << std::endl;
    for (int i = 0; i < max_elements; i++) {
        auto result = index->searchKnn(data + i * dim, k, lid_threshold);
        
        // Check if point i appears in any of the k results
        bool found = false;
        std::vector<std::pair<float, hnswlib::labeltype>> temp_results;
        while (!result.empty()) {
            if (result.top().second == i) {
                found = true;
                break;
            }
            temp_results.push_back(result.top());
            result.pop();
        }
        if (found) correct++;
    }

    recall = (float)correct / max_elements;
    std::cout << "\nDeserialized index:" << std::endl;
    std::cout << "k value: " << k << std::endl;
    std::cout << "Recall@" << k << ": " << recall << "\n";

    delete[] data;
    delete index;
    return 0;
}