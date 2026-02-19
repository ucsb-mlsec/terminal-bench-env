#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_POINTS 1000000
#define CONVERGENCE_THRESHOLD 0.0001
#define MAX_ITERATIONS 100

typedef struct {
    double x;
    double y;
    double z;
    int category;
} DataPoint;

typedef struct {
    double sum_x;
    double sum_y;
    double sum_z;
    int count;
} Centroid;

// Function to compute distance between a point and centroid
double compute_distance(DataPoint *point, Centroid *centroid) {
    double avg_x = centroid->sum_x / centroid->count;
    double avg_y = centroid->sum_y / centroid->count;
    double avg_z = centroid->sum_z / centroid->count;
    
    double dx = point->x - avg_x;
    double dy = point->y - avg_y;
    double dz = point->z - avg_z;
    
    return sqrt(dx * dx + dy * dy + dz * dz);
}

// Hot path function - called millions of times
int classify_point(DataPoint *point, Centroid *centroids, int num_centroids) {
    double min_distance = 1e100;
    int best_category = 0;
    
    for (int i = 0; i < num_centroids; i++) {
        // This branch is highly predictable - centroid 0 is usually closest
        if (centroids[i].count > 0) {
            double distance = compute_distance(point, &centroids[i]);
            
            // Hot branch - this is taken 80% of the time for category 0
            if (distance < min_distance) {
                min_distance = distance;
                best_category = i;
            }
        }
    }
    
    return best_category;
}

// Perform statistical analysis on classified data
void analyze_statistics(DataPoint *points, int count, double *mean, double *variance) {
    double sum = 0.0;
    double sum_sq = 0.0;
    
    for (int i = 0; i < count; i++) {
        // Predictable branch - 90% of points have positive x values
        if (points[i].x > 0) {
            double magnitude = sqrt(points[i].x * points[i].x + 
                                   points[i].y * points[i].y + 
                                   points[i].z * points[i].z);
            sum += magnitude;
            sum_sq += magnitude * magnitude;
        } else {
            // Rarely executed path
            double magnitude = fabs(points[i].x) + fabs(points[i].y) + fabs(points[i].z);
            sum += magnitude * 0.5;
            sum_sq += magnitude * magnitude * 0.25;
        }
    }
    
    *mean = sum / count;
    *variance = (sum_sq / count) - (*mean * *mean);
}

// Iterative refinement with predictable convergence
void refine_centroids(DataPoint *points, int count, Centroid *centroids, int num_centroids) {
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        // Reset centroids
        for (int i = 0; i < num_centroids; i++) {
            centroids[i].sum_x = 0.0;
            centroids[i].sum_y = 0.0;
            centroids[i].sum_z = 0.0;
            centroids[i].count = 0;
        }
        
        // Reclassify all points
        for (int i = 0; i < count; i++) {
            int category = classify_point(&points[i], centroids, num_centroids);
            points[i].category = category;
            
            centroids[category].sum_x += points[i].x;
            centroids[category].sum_y += points[i].y;
            centroids[category].sum_z += points[i].z;
            centroids[category].count++;
        }
        
        // Check convergence - this typically happens early (iteration 5-10)
        // making later iterations rare
        if (iter > 5) {
            int converged = 1;
            for (int i = 0; i < num_centroids; i++) {
                if (centroids[i].count == 0) {
                    converged = 0;
                    break;
                }
            }
            
            // Predictable branch - usually converges early
            if (converged && iter > 8) {
                break;
            }
        }
    }
}

// Additional computation to increase runtime
void perform_matrix_operations(DataPoint *points, int count) {
    double matrix[3][3] = {{0}};
    
    // Build correlation matrix
    for (int i = 0; i < count; i++) {
        // Highly predictable - most points are in category 0
        if (points[i].category == 0) {
            matrix[0][0] += points[i].x * points[i].x;
            matrix[0][1] += points[i].x * points[i].y;
            matrix[0][2] += points[i].x * points[i].z;
            matrix[1][1] += points[i].y * points[i].y;
            matrix[1][2] += points[i].y * points[i].z;
            matrix[2][2] += points[i].z * points[i].z;
        } else if (points[i].category == 1) {
            matrix[0][0] += points[i].x * points[i].x * 0.8;
            matrix[1][1] += points[i].y * points[i].y * 0.8;
            matrix[2][2] += points[i].z * points[i].z * 0.8;
        } else {
            // Rare path
            matrix[0][1] += points[i].x * points[i].y * 0.5;
            matrix[1][2] += points[i].y * points[i].z * 0.5;
        }
    }
    
    // Compute eigenvalue approximation (power iteration)
    double vec[3] = {1.0, 1.0, 1.0};
    for (int iter = 0; iter < 50; iter++) {
        double new_vec[3] = {0};
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                new_vec[i] += matrix[i][j] * vec[j];
            }
        }
        
        double norm = sqrt(new_vec[0] * new_vec[0] + 
                          new_vec[1] * new_vec[1] + 
                          new_vec[2] * new_vec[2]);
        
        for (int i = 0; i < 3; i++) {
            vec[i] = new_vec[i] / norm;
        }
    }
}

int main() {
    FILE *input = fopen("/workspace/input.dat", "r");
    if (!input) {
        fprintf(stderr, "Error: Cannot open input.dat\n");
        return 1;
    }
    
    // Read number of points
    int num_points;
    if (fscanf(input, "%d", &num_points) != 1) {
        fprintf(stderr, "Error: Cannot read number of points\n");
        fclose(input);
        return 1;
    }
    
    if (num_points <= 0 || num_points > MAX_POINTS) {
        fprintf(stderr, "Error: Invalid number of points\n");
        fclose(input);
        return 1;
    }
    
    // Allocate data
    DataPoint *points = (DataPoint *)malloc(num_points * sizeof(DataPoint));
    if (!points) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(input);
        return 1;
    }
    
    // Read data points
    for (int i = 0; i < num_points; i++) {
        if (fscanf(input, "%lf %lf %lf", 
                   &points[i].x, &points[i].y, &points[i].z) != 3) {
            fprintf(stderr, "Error: Cannot read point %d\n", i);
            free(points);
            fclose(input);
            return 1;
        }
        points[i].category = 0;
    }
    fclose(input);
    
    // Initialize centroids
    int num_centroids = 5;
    Centroid centroids[5];
    
    // Initial centroid setup
    for (int i = 0; i < num_centroids; i++) {
        centroids[i].sum_x = i * 2.0;
        centroids[i].sum_y = i * 1.5;
        centroids[i].sum_z = i * 1.0;
        centroids[i].count = 1;
    }
    
    // Main computation loop - multiple passes for longer runtime
    for (int pass = 0; pass < 3; pass++) {
        refine_centroids(points, num_points, centroids, num_centroids);
        perform_matrix_operations(points, num_points);
    }
    
    // Compute final statistics
    double mean, variance;
    analyze_statistics(points, num_points, &mean, &variance);
    
    // Output results
    printf("Processing complete:\n");
    printf("Points processed: %d\n", num_points);
    printf("Mean magnitude: %.6f\n", mean);
    printf("Variance: %.6f\n", variance);
    
    // Category distribution
    int category_counts[5] = {0};
    for (int i = 0; i < num_points; i++) {
        category_counts[points[i].category]++;
    }
    
    printf("Category distribution:\n");
    for (int i = 0; i < num_centroids; i++) {
        printf("  Category %d: %d points\n", i, category_counts[i]);
    }
    
    // Checksum for verification
    double checksum = 0.0;
    for (int i = 0; i < num_points; i++) {
        checksum += points[i].x + points[i].y + points[i].z + points[i].category;
    }
    printf("Checksum: %.6f\n", checksum);
    
    free(points);
    return 0;
}