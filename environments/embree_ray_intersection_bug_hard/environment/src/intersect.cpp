#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

const double EPSILON = 1e-6;

struct Vec3 {
    double x, y, z;
    
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}
};

struct Ray {
    Vec3 origin;
    Vec3 direction;
};

struct Triangle {
    Vec3 v0, v1, v2;
};

struct TestCase {
    Ray ray;
    Triangle tri;
    bool expect_hit;
    Vec3 expect_point;
};

Vec3 subtract(const Vec3& a, const Vec3& b) {
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

bool intersectRayTriangle(const Ray& ray, const Triangle& tri, Vec3& hit_point) {
    // Möller-Trumbore algorithm
    Vec3 edge1 = subtract(tri.v1, tri.v0);
    Vec3 edge2 = subtract(tri.v2, tri.v0);
    
    Vec3 h = cross(ray.direction, edge2);
    double a = dot(edge1, h);
    
    if (a > -EPSILON && a < EPSILON) {
        return false; // Ray is parallel to triangle
    }
    
    double f = 1.0 / a;
    Vec3 s = subtract(ray.origin, tri.v0);
    double u = f * dot(s, h);
    
    if (u < 0.0 || u > 1.0) {
        return false;
    }
    
    Vec3 q = cross(s, edge1);
    double v = f * dot(ray.direction, q);
    
    // BUG: Sign error introduced here
    if (v < 0.0 || u - v > 1.0) {
        return false;
    }
    
    double t = f * dot(edge2, q);
    
    if (t > EPSILON) {
        hit_point.x = ray.origin.x + ray.direction.x * t;
        hit_point.y = ray.origin.y + ray.direction.y * t;
        hit_point.z = ray.origin.z + ray.direction.z * t;
        return true;
    }
    
    return false;
}

std::vector<TestCase> loadTestCases(const std::string& filename) {
    std::vector<TestCase> cases;
    std::ifstream file(filename);
    std::string line;
    
    TestCase current;
    int lineNum = 0;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;
        
        if (type == "RAY") {
            iss >> current.ray.origin.x >> current.ray.origin.y >> current.ray.origin.z
                >> current.ray.direction.x >> current.ray.direction.y >> current.ray.direction.z;
            lineNum = 1;
        } else if (type == "TRI") {
            iss >> current.tri.v0.x >> current.tri.v0.y >> current.tri.v0.z
                >> current.tri.v1.x >> current.tri.v1.y >> current.tri.v1.z
                >> current.tri.v2.x >> current.tri.v2.y >> current.tri.v2.z;
            lineNum = 2;
        } else if (type == "EXPECT") {
            std::string hit_str;
            iss >> hit_str;
            current.expect_hit = (hit_str == "true");
            if (current.expect_hit) {
                iss >> current.expect_point.x >> current.expect_point.y >> current.expect_point.z;
            }
            cases.push_back(current);
            lineNum = 0;
        }
    }
    
    return cases;
}

bool pointsEqual(const Vec3& a, const Vec3& b, double epsilon = 0.001) {
    return std::abs(a.x - b.x) < epsilon &&
           std::abs(a.y - b.y) < epsilon &&
           std::abs(a.z - b.z) < epsilon;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test_cases_file>" << std::endl;
        return 1;
    }
    
    std::vector<TestCase> testCases = loadTestCases(argv[1]);
    
    int passed = 0;
    int total = testCases.size();
    
    for (size_t i = 0; i < testCases.size(); i++) {
        const TestCase& tc = testCases[i];
        Vec3 hit_point;
        bool hit = intersectRayTriangle(tc.ray, tc.tri, hit_point);
        
        bool test_passed = false;
        
        if (hit == tc.expect_hit) {
            if (!hit) {
                test_passed = true;
            } else if (pointsEqual(hit_point, tc.expect_point)) {
                test_passed = true;
            }
        }
        
        if (test_passed) {
            std::cout << "Test " << (i + 1) << ": PASS" << std::endl;
            passed++;
        } else {
            std::cout << "Test " << (i + 1) << ": FAIL (expected hit=" 
                      << (tc.expect_hit ? "true" : "false") 
                      << " got hit=" << (hit ? "true" : "false") << ")";
            if (hit && tc.expect_hit) {
                std::cout << " point mismatch: expected (" 
                          << tc.expect_point.x << "," << tc.expect_point.y << "," << tc.expect_point.z
                          << ") got (" << hit_point.x << "," << hit_point.y << "," << hit_point.z << ")";
            }
            std::cout << std::endl;
        }
    }
    
    std::cout << std::endl << passed << "/" << total << " tests passed" << std::endl;
    
    return (passed == total) ? 0 : 1;
}