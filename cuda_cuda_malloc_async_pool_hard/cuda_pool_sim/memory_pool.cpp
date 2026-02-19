#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <cstring>
#include <cassert>

class MemoryPool {
private:
    std::map<void*, size_t> allocated_blocks;
    size_t total_allocated;
    size_t total_freed;

public:
    MemoryPool() : total_allocated(0), total_freed(0) {}

    ~MemoryPool() {
        // Clean up any remaining allocations
        for (auto& pair : allocated_blocks) {
            free(pair.first);
        }
        allocated_blocks.clear();
    }

    void* allocate(size_t size) {
        void* ptr = malloc(size);
        
        // Bug 3: Not checking if malloc returned nullptr before tracking
        allocated_blocks[ptr] = size;
        total_allocated += size;
        
        return ptr;
    }

    void deallocate(void* ptr) {
        if (ptr == nullptr) {
            return;
        }

        auto it = allocated_blocks.find(ptr);
        if (it == allocated_blocks.end()) {
            std::cerr << "ERROR: Attempting to free untracked pointer" << std::endl;
            return;
        }

        // Bug 1: Accessing allocated_blocks[ptr] after free()
        free(ptr);
        size_t block_size = allocated_blocks[ptr];
        
        // Bug 2: Incrementing by hardcoded value instead of actual block size
        total_freed += 1024;
        
        allocated_blocks.erase(ptr);
    }

    void get_stats(size_t& alloc, size_t& freed, size_t& active) {
        alloc = total_allocated;
        freed = total_freed;
        active = allocated_blocks.size();
    }

    bool verify_integrity() {
        size_t calculated_allocated = 0;
        for (const auto& pair : allocated_blocks) {
            calculated_allocated += pair.second;
        }
        
        size_t expected_active = total_allocated - total_freed;
        
        if (calculated_allocated != expected_active) {
            std::cerr << "INTEGRITY ERROR: Calculated active memory (" 
                      << calculated_allocated << ") != Expected (" 
                      << expected_active << ")" << std::endl;
            return false;
        }
        
        return true;
    }

    bool has_allocation(void* ptr) {
        return allocated_blocks.find(ptr) != allocated_blocks.end();
    }

    size_t get_active_blocks() {
        return allocated_blocks.size();
    }
};

bool validate_memory_pattern(void* ptr, size_t size, unsigned char pattern) {
    unsigned char* data = static_cast<unsigned char*>(ptr);
    
    // Bug 4: Off-by-one error in validation loop
    for (size_t i = 0; i <= size; i++) {
        if (data[i] != pattern) {
            std::cerr << "VALIDATION ERROR: Memory pattern mismatch at offset " << i << std::endl;
            return false;
        }
    }
    return true;
}

void write_memory_pattern(void* ptr, size_t size, unsigned char pattern) {
    unsigned char* data = static_cast<unsigned char*>(ptr);
    memset(data, pattern, size);
}

int main() {
    std::cout << "Starting CUDA Memory Pool Simulation" << std::endl;
    std::cout << "Target operations: 5000" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    MemoryPool pool;
    std::vector<void*> active_allocations;
    const int TOTAL_OPERATIONS = 5000;
    const size_t MIN_ALLOC_SIZE = 64;
    const size_t MAX_ALLOC_SIZE = 4096;
    
    int operations_completed = 0;
    int allocations_made = 0;
    int deallocations_made = 0;

    srand(42); // Fixed seed for reproducibility

    for (int op = 0; op < TOTAL_OPERATIONS; op++) {
        // Decide whether to allocate or deallocate
        bool should_allocate = active_allocations.empty() || 
                              (active_allocations.size() < 100 && (rand() % 100) < 60);

        if (should_allocate) {
            // Allocate memory
            size_t size = MIN_ALLOC_SIZE + (rand() % (MAX_ALLOC_SIZE - MIN_ALLOC_SIZE));
            void* ptr = pool.allocate(size);
            
            if (ptr == nullptr) {
                std::cerr << "ERROR: Allocation failed at operation " << op << std::endl;
                break;
            }

            // Write a pattern to the memory
            unsigned char pattern = static_cast<unsigned char>(rand() % 256);
            write_memory_pattern(ptr, size, pattern);

            active_allocations.push_back(ptr);
            allocations_made++;
            
        } else if (!active_allocations.empty()) {
            // Deallocate memory
            size_t index = rand() % active_allocations.size();
            void* ptr = active_allocations[index];
            
            // Validate memory pattern before freeing
            if (!pool.has_allocation(ptr)) {
                std::cerr << "ERROR: Pointer not tracked by pool" << std::endl;
                break;
            }

            pool.deallocate(ptr);
            
            // Remove from active list
            active_allocations.erase(active_allocations.begin() + index);
            deallocations_made++;
        }

        operations_completed++;

        // Progress reporting
        if ((operations_completed % 1000) == 0) {
            size_t alloc, freed, active;
            pool.get_stats(alloc, freed, active);
            std::cout << "Progress: " << operations_completed << " operations | "
                      << "Allocations: " << allocations_made << " | "
                      << "Deallocations: " << deallocations_made << " | "
                      << "Active blocks: " << active << std::endl;
            
            if (!pool.verify_integrity()) {
                std::cerr << "FATAL: Integrity check failed at operation " 
                          << operations_completed << std::endl;
                break;
            }
        }
    }

    // Clean up remaining allocations
    std::cout << "\nCleaning up remaining allocations..." << std::endl;
    for (void* ptr : active_allocations) {
        pool.deallocate(ptr);
        deallocations_made++;
    }
    active_allocations.clear();

    // Final statistics
    std::cout << "\n========================================" << std::endl;
    std::cout << "Final Statistics:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Operations completed: " << operations_completed << " / " << TOTAL_OPERATIONS << std::endl;
    std::cout << "Total allocations: " << allocations_made << std::endl;
    std::cout << "Total deallocations: " << deallocations_made << std::endl;
    
    size_t final_alloc, final_freed, final_active;
    pool.get_stats(final_alloc, final_freed, final_active);
    
    std::cout << "Total memory allocated: " << final_alloc << " bytes" << std::endl;
    std::cout << "Total memory freed: " << final_freed << " bytes" << std::endl;
    std::cout << "Active blocks remaining: " << final_active << std::endl;

    // Final integrity check
    bool success = true;
    if (operations_completed != TOTAL_OPERATIONS) {
        std::cout << "\nSTATUS: FAIL - Did not complete all operations" << std::endl;
        success = false;
    } else if (final_active != 0) {
        std::cout << "\nSTATUS: FAIL - Memory leak detected (" << final_active << " blocks)" << std::endl;
        success = false;
    } else if (final_alloc != final_freed) {
        std::cout << "\nSTATUS: FAIL - Memory accounting mismatch" << std::endl;
        success = false;
    } else if (!pool.verify_integrity()) {
        std::cout << "\nSTATUS: FAIL - Integrity check failed" << std::endl;
        success = false;
    } else {
        std::cout << "\nSTATUS: PASS - All operations completed successfully" << std::endl;
    }

    std::cout << "========================================" << std::endl;

    return success ? 0 : 1;
}