#include <cuda_runtime.h>

struct GraphNode {
    int id;
    int numNeighbors;
    int neighbors[8];
    int data;
};

// Entry point: process_graph_kernel
__global__ void explore_deep_kernel(GraphNode* graph, int nodeId, int remainingDepth);
__global__ void process_neighbor_kernel(GraphNode* graph, int* visited, int nodeId, int depth);

__global__ void process_graph_kernel(GraphNode* graph, int* visited, int nodeId, int depth) {
    if (nodeId < 0 || depth > 30) return;
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Mark current node as visited
    if (tid == 0) {
        visited[nodeId] = 1;
    }
    __syncthreads();
    
    GraphNode node = graph[nodeId];
    
    // Process all neighbors
    for (int i = 0; i < node.numNeighbors; i++) {
        int neighborId = node.neighbors[i];
        
        if (visited[neighborId] == 0) {
            // Launch kernel for unvisited neighbor
            if (i % 2 == 0) {
                process_neighbor_kernel<<<1, 32>>>(graph, visited, neighborId, depth + 1);
            }
        }
        
        // Additional processing for high-priority neighbors
        if (node.data > 100 && i < 2) {
            process_neighbor_kernel<<<1, 16>>>(graph, visited, neighborId, depth + 1);
        }
    }
    
    // Synchronize all child kernels
    cudaDeviceSynchronize();
}

__global__ void process_neighbor_kernel(GraphNode* graph, int* visited, int nodeId, int depth) {
    if (nodeId < 0 || depth > 30) return;
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    GraphNode node = graph[nodeId];
    
    // Mark as visited
    if (tid == 0 && visited[nodeId] == 0) {
        visited[nodeId] = 1;
    }
    __syncthreads();
    
    // Process node data
    int processedData = node.data * 2;
    
    // Recursive exploration for deeper levels
    if (depth < 10) {
        for (int i = 0; i < node.numNeighbors; i++) {
            int neighborId = node.neighbors[i];
            
            if (visited[neighborId] == 0) {
                // Recursive call to continue traversal
                process_neighbor_kernel<<<1, 32>>>(graph, visited, neighborId, depth + 1);
            }
        }
        
        // Explore deep paths for complex nodes
        if (node.numNeighbors > 4 && depth < 8) {
            explore_deep_kernel<<<1, 16>>>(graph, nodeId, 18);
        }
    }
    
    // Additional boundary processing
    if (depth == 9 && node.data > 50) {
        for (int i = 0; i < node.numNeighbors; i++) {
            int neighborId = node.neighbors[i];
            if (neighborId % 3 == 0) {
                explore_deep_kernel<<<1, 8>>>(graph, neighborId, 15);
            }
        }
    }
    
    cudaDeviceSynchronize();
}

__global__ void explore_deep_kernel(GraphNode* graph, int nodeId, int remainingDepth) {
    if (nodeId < 0 || remainingDepth <= 0) return;
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    GraphNode node = graph[nodeId];
    
    // Complex exploration logic
    int explorationFactor = node.data % 4;
    
    // Path 1: Deep sequential exploration
    if (explorationFactor == 0 && remainingDepth > 1) {
        for (int i = 0; i < node.numNeighbors && i < 2; i++) {
            int neighborId = node.neighbors[i];
            explore_deep_kernel<<<1, 8>>>(graph, neighborId, remainingDepth - 1);
        }
    }
    
    // Path 2: Wide exploration with depth
    if (explorationFactor == 1 && remainingDepth > 2) {
        for (int i = 0; i < node.numNeighbors; i++) {
            int neighborId = node.neighbors[i];
            explore_deep_kernel<<<1, 4>>>(graph, neighborId, remainingDepth - 2);
        }
    }
    
    // Path 3: Conditional deep dive
    if (node.data > 75 && remainingDepth > 3) {
        int primaryNeighbor = node.neighbors[0];
        if (primaryNeighbor >= 0) {
            explore_deep_kernel<<<1, 8>>>(graph, primaryNeighbor, remainingDepth - 1);
        }
        
        // Secondary exploration
        if (node.numNeighbors > 2) {
            int secondaryNeighbor = node.neighbors[1];
            explore_deep_kernel<<<1, 4>>>(graph, secondaryNeighbor, remainingDepth - 1);
        }
    }
    
    // Path 4: Fallback exploration
    if (remainingDepth > 4 && explorationFactor >= 2) {
        for (int i = 0; i < node.numNeighbors && i < 3; i++) {
            if (node.neighbors[i] % 2 == 1) {
                explore_deep_kernel<<<1, 8>>>(graph, node.neighbors[i], remainingDepth - 1);
            }
        }
    }
    
    __syncthreads();
    
    // Final boundary check
    if (remainingDepth == 1 && tid == 0) {
        // Leaf processing
        node.data += 1;
    }
    
    cudaDeviceSynchronize();
}

// Helper device function for data processing
__device__ int processNodeData(GraphNode* node) {
    return node->data * node->numNeighbors;
}

// Utility kernel for initialization
__global__ void initialize_visited(int* visited, int numNodes) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numNodes) {
        visited[tid] = 0;
    }
}