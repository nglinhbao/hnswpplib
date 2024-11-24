import hnswpplib
import numpy as np
import pickle

"""
Example of search using HNSWPP
"""

def main():
    dim = 128
    num_elements = 10000

    # Generating sample data
    data = np.float32(np.random.random((num_elements, dim)))
    ids = np.arange(num_elements)

    # Declaring index - Note: passing arguments positionally
    p = hnswpplib.Index('l2', dim)  # possible options are l2, cosine or ip

    # Initializing index with HNSWPP-specific parameters
    p.init_index(
        max_elements=num_elements,
        M=16,
        ef_construction=200,
        random_seed=100,
        allow_replace_deleted=False
    )

    # Element insertion (requires initial data for LID computation)
    p.prepare_index(data)  # Prepare index with data for LID computation
    p.add_items(data, ids)

    # Query dataset, k - number of the closest elements (returns 2 numpy arrays)
    labels, distances = p.knn_query(data, k=1)

    print(f"Parameters passed to constructor: space={p.space}, dim={p.dim}")
    print(f"Index size is {len(labels)}")
    print(f"First 5 nearest neighbors and distances:")
    for i in range(5):
        print(f"Query {i}: Nearest neighbor: {labels[i][0]}, Distance: {distances[i][0]}")

    # Verify results
    print("\nVerifying results...")
    errors = 0
    for i in range(num_elements):
        if labels[i][0] != i:
            errors += 1
            if errors < 5:  # Print first 5 errors only
                print(f"Error at {i}: Retrieved {labels[i][0]} instead")
    print(f"Total errors: {errors}")
    
    # Test save/load functionality
    print("\nTesting save/load...")
    # Save to temporary files
    temp_path = "temp_index"
    p.save_index(temp_path)
    
    # Create new index
    p_loaded = hnswpplib.Index('l2', dim)
    p_loaded.load_index(temp_path, max_elements=num_elements)
    
    # Query loaded index
    labels_loaded, distances_loaded = p_loaded.knn_query(data, k=1)
    
    # Verify loaded results
    errors_loaded = 0
    for i in range(num_elements):
        if labels_loaded[i][0] != labels[i][0]:
            errors_loaded += 1
    
    print(f"Errors after load: {errors_loaded}")

if __name__ == '__main__':
    main()