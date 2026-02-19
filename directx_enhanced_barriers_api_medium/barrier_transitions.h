#ifndef BARRIER_TRANSITIONS_H
#define BARRIER_TRANSITIONS_H

#include <d3d12.h>
#include <vector>
#include <string>

namespace dx12 {

/**
 * DirectX 12 Enhanced Barrier API Rules:
 * 
 * 1. COMMON state transitions:
 *    - COMMON is a special state that can transition to/from most states
 *    - Transitioning COMMON -> COMMON is redundant and invalid
 *    - Some state pairs require intermediate COMMON transition
 * 
 * 2. State compatibility:
 *    - RENDER_TARGET, DEPTH_WRITE, UNORDERED_ACCESS are exclusive write states
 *    - Cannot directly transition between incompatible write states
 *    - DEPTH_READ and DEPTH_WRITE require exclusivity (no simultaneous access)
 * 
 * 3. Resource type constraints:
 *    - DEPTH_READ/DEPTH_WRITE only valid for DEPTH_STENCIL resources
 *    - RENDER_TARGET typically for TEXTURE_2D resources
 *    - VERTEX_BUFFER/INDEX_BUFFER only for BUFFER resources
 * 
 * 4. Synchronization scope:
 *    - UNORDERED_ACCESS requires COMPUTE or GRAPHICS scope
 *    - PRESENT requires GRAPHICS scope
 *    - Mismatched scopes can cause synchronization issues
 */

// Resource states in DirectX 12 enhanced barrier model
enum class ResourceState {
    COMMON,
    VERTEX_BUFFER,
    INDEX_BUFFER,
    CONSTANT_BUFFER,
    RENDER_TARGET,
    DEPTH_READ,
    DEPTH_WRITE,
    UNORDERED_ACCESS,
    SHADER_RESOURCE,
    COPY_SOURCE,
    COPY_DEST,
    PRESENT,
    NON_PIXEL_SHADER_RESOURCE,
    PIXEL_SHADER_RESOURCE
};

// Resource types that can have barriers
enum class ResourceType {
    BUFFER,
    TEXTURE_2D,
    TEXTURE_3D,
    DEPTH_STENCIL
};

// Synchronization scopes for barrier execution
enum class SyncScope {
    GLOBAL,
    COMPUTE,
    GRAPHICS,
    COPY
};

// Structure representing a barrier transition
struct BarrierTransition {
    std::string id;
    ResourceState srcState;
    ResourceState dstState;
    ResourceType resourceType;
    SyncScope syncScope;
    
    BarrierTransition() = default;
    BarrierTransition(const std::string& barrierId, ResourceState src, ResourceState dst,
                     ResourceType type, SyncScope scope)
        : id(barrierId), srcState(src), dstState(dst), resourceType(type), syncScope(scope) {}
};

// Validation functions

/**
 * Validates if a barrier transition is valid according to DirectX 12 enhanced barrier rules
 * @param barrier The barrier transition to validate
 * @return true if the transition is valid, false otherwise
 */
bool IsValidTransition(const BarrierTransition& barrier);

/**
 * Checks if two resource states are compatible for direct transition
 * @param src Source resource state
 * @param dst Destination resource state
 * @return true if states can transition directly without intermediate COMMON state
 */
bool AreStatesCompatible(ResourceState src, ResourceState dst);

/**
 * Validates if the resource type is compatible with the given state
 * @param state The resource state
 * @param type The resource type
 * @return true if the resource type can be in the specified state
 */
bool IsStateValidForResourceType(ResourceState state, ResourceType type);

/**
 * Checks if the synchronization scope is appropriate for the barrier
 * @param barrier The barrier transition
 * @return true if the sync scope is valid for the state transition
 */
bool IsSyncScopeValid(const BarrierTransition& barrier);

/**
 * Determines if a state is an exclusive write state
 * @param state The resource state to check
 * @return true if the state requires exclusive write access
 */
bool IsExclusiveWriteState(ResourceState state);

/**
 * Validates depth resource specific rules
 * @param barrier The barrier transition for depth resource
 * @return true if depth-specific rules are satisfied
 */
bool ValidateDepthTransition(const BarrierTransition& barrier);

} // namespace dx12

#endif // BARRIER_TRANSITIONS_H