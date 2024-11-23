#include "../../src/hnswlib.h"


int main() {
    int dim = 16;               // Dimension of the elements
    int max_elements = 1000;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i * dim, i, -1);
    }

    int k = 10;
    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, k);
        
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
    std::string hnsw_path = "hnsw.bin";
    alg_hnsw->saveIndex(hnsw_path);
    delete alg_hnsw;

    // Deserialize index and check recall
    alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
    correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, k);
        
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
    delete alg_hnsw;
    return 0;
}