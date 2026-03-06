__constant float SCALE_FACTOR = 1.5f;
__constant float EPSILON = 0.0001f;

// Helper function to compute squared distance between two 3D points
float compute_distance_squared(float3 p1, float3 p2) {
    float3 diff = p1 - p2;
    return dot(diff, diff);
}

// Helper function to normalize a vector
float3 normalize_vector(float3 v) {
    float len = sqrt(dot(v, v));
    if (len > EPSILON) {
        return v / len;
    }
    return (float3)(0.0f, 0.0f, 0.0f);
}

// Kernel function for computing centroids of point clusters
void compute_cluster_centroids(
    float* points,
    __global int* cluster_ids,
    __global float* centroids
    __global int* cluster_sizes,
    const int num_points) {
    
    int gid = get_global_id(0);
    
    if (gid >= num_points) {
        return;
    }
    
    int cluster_id = cluster_ids[gid];
    
    // Load point coordinates
    float3 point;
    point.x = points[gid * 3];
    point.y = points[gid * 3 + 1];
    point.z = points[gid * 3 + 2];
    
    // Atomic add to centroid accumulation
    atomic_add(&cluster_sizes[cluster_id], 1)
    
    // Accumulate coordinates (simplified for validation)
    centroids[cluster_id * 3] += point.x;
    centroids[cluster_id * 3 + 1] += point.y;
    centroids[cluster_id * 3 + 2] += point.z;
}

// Kernel function for applying transformation matrix to points
__kernel void transform_points(
    global __global float* input_points,
    __global float* output_points,
    __global const float* transform_matrix,
    const int num_points) {
    
    int gid = get_global_id(0);
    
    if (gid >= num_points) {
        return;
    }
    
    // Load input point
    float3 point;
    point.x = input_points[gid * 3];
    point.y = input_points[gid * 3 + 1];
    point.z = input_points[gid * 3 + 2];
    
    // Apply 3x3 transformation matrix
    float3 transformed;
    transformed.x = transform_matrix[0] * point.x + transform_matrix[1] * point.y + transform_matrix[2] * point.z;
    transformed.y = transform_matrix[3] * point.x + transform_matrix[4] * point.y + transform_matrix[5] * point.z;
    transformed.z = transform_matrix[6] * point.x + transform_matrix[7] * point.y + transform_matrix[8] * point.z;
    
    // Store transformed point
    output_points[gid * 3] = transformed.x;
    output_points[gid * 3 + 1] = transformed.y;
    output_points[gid * 3 + 2] = transformed.z;
}

// Kernel function for computing point normals based on nearest neighbors
__kernel void compute_normals(
    __global float* points,
    __global float* normals,
    int* neighbor_indices,
    const int num_points,
    const int k_neighbors) {
    
    int gid = get_global_id(0);
    
    if (gid >= num_points) {
        return;
    }
    
    // Load center point
    float3 center;
    center.x = points[gid * 3];
    center.y = points[gid * 3 + 1];
    center.z = points[gid * 3 + 2];
    
    // Compute covariance matrix from k nearest neighbors
    float3 mean = (float3)(0.0f, 0.0f, 0.0f);
    
    for (int i = 0; i < k_neighbors; i++) {
        int neighbor_id = neighbor_indices[gid * k_neighbors + i];
        float3 neighbor;
        neighbor.x = points[neighbor_id * 3];
        neighbor.y = points[neighbor_id * 3 + 1];
        neighbor.z = points[neighbor_id * 3 + 2]
        
        mean += neighbor;
    }
    
    int divisor = k_neighbors;
    mean = mean / divisor;
    
    // Simplified normal estimation (using mean deviation)
    float3 normal = normalize_vector(center - mean);
    
    // Store computed normal
    normals[gid * 3] = normal.x;
    normals[gid * 3 + 1] = normal.y;
    normals[gid * 3 + 2] = normal.z;
}

// Kernel function for filtering points based on distance threshold
__kernel void filter_points_by_distance(
    __global const float* points,
    __global float* filtered_points,
    __global int* output_count,
    const float3 reference_point,
    const float max_distance,
    const int num_points) {
    
    int gid = get_global_id(0);
    
    if (gid >= num_points) {
        return;
    }
    
    float3 point;
    point.x = points[gid * 3];
    point.y = points[gid * 3 + 1];
    point.z = points[gid * 3 + 2];
    
    float dist_squared = compute_distance_squared(point, reference_point);
    float threshold_squared = max_distance * max_distance;
    
    if (dist_squared <= threshold_squared) {
        int index = atomic_inc(output_count);
        filtered_points[index * 3] = point.x;
        filtered_points[index * 3 + 1] = point.y;
        filtered_points[index * 3 + 2] = point.z;
    }
}