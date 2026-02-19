#!/usr/bin/env python3

import json
import os

def parse_obj(filename):
    """Parse OBJ file and return vertices and faces."""
    vertices = []
    faces = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.split()
                # OBJ indices start at 1, convert to 0-based
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face)
    
    return vertices, faces

def build_edge_face_map(faces):
    """Build a map of edges to their adjacent faces."""
    edge_faces = {}
    
    for face_idx, face in enumerate(faces):
        for i in range(len(face)):
            v1 = face[i]
            v2 = face[(i + 1) % len(face)]
            # Normalize edge representation (smaller index first)
            edge = tuple(sorted([v1, v2]))
            
            if edge not in edge_faces:
                edge_faces[edge] = []
            edge_faces[edge].append(face_idx)
    
    return edge_faces

def calculate_face_points(vertices, faces):
    """Calculate face points (centroid of each face). BUG: Incorrect averaging."""
    face_points = []
    
    for face in faces:
        # BUG: Missing division by number of vertices in face
        centroid = [0.0, 0.0, 0.0]
        for v_idx in face:
            centroid[0] += vertices[v_idx][0]
            centroid[1] += vertices[v_idx][1]
            centroid[2] += vertices[v_idx][2]
        # BUG: Should divide by len(face), but we forget to do it
        face_points.append(centroid)
    
    return face_points

def calculate_edge_points(vertices, faces, edge_faces, face_points):
    """Calculate edge points. BUG: Incorrect formula."""
    edge_points = {}
    
    for edge, adjacent_faces in edge_faces.items():
        v1, v2 = edge
        
        # Edge midpoint
        midpoint = [
            (vertices[v1][0] + vertices[v2][0]) / 2.0,
            (vertices[v1][1] + vertices[v2][1]) / 2.0,
            (vertices[v1][2] + vertices[v2][2]) / 2.0
        ]
        
        # Average of adjacent face points
        face_avg = [0.0, 0.0, 0.0]
        for face_idx in adjacent_faces:
            face_avg[0] += face_points[face_idx][0]
            face_avg[1] += face_points[face_idx][1]
            face_avg[2] += face_points[face_idx][2]
        
        # BUG: Wrong formula - should be (midpoint + face_avg) / 2, but we use wrong weights
        edge_point = [
            (midpoint[0] + face_avg[0]) / 2.0,  # BUG: Missing division by len(adjacent_faces) for face_avg
            (midpoint[1] + face_avg[1]) / 2.0,
            (midpoint[2] + face_avg[2]) / 2.0
        ]
        
        edge_points[edge] = edge_point
    
    return edge_points

def reposition_original_vertices(vertices, faces, edge_faces, face_points, edge_points):
    """Reposition original vertices using Catmull-Clark formula. BUG: Wrong formula."""
    new_positions = []
    
    for v_idx in range(len(vertices)):
        # Find all faces containing this vertex
        adjacent_faces = []
        adjacent_edges = []
        
        for face_idx, face in enumerate(faces):
            if v_idx in face:
                adjacent_faces.append(face_idx)
        
        for edge in edge_faces.keys():
            if v_idx in edge:
                adjacent_edges.append(edge)
        
        n = len(adjacent_faces)  # valence
        
        if n == 0:
            new_positions.append(vertices[v_idx])
            continue
        
        # Calculate F (average of adjacent face points)
        F = [0.0, 0.0, 0.0]
        for face_idx in adjacent_faces:
            F[0] += face_points[face_idx][0]
            F[1] += face_points[face_idx][1]
            F[2] += face_points[face_idx][2]
        F = [F[0] / n, F[1] / n, F[2] / n]
        
        # Calculate R (average of adjacent edge midpoints)
        R = [0.0, 0.0, 0.0]
        for edge in adjacent_edges:
            v1, v2 = edge
            R[0] += (vertices[v1][0] + vertices[v2][0]) / 2.0
            R[1] += (vertices[v1][1] + vertices[v2][1]) / 2.0
            R[2] += (vertices[v1][2] + vertices[v2][2]) / 2.0
        R = [R[0] / n, R[1] / n, R[2] / n]
        
        # Original vertex position
        P = vertices[v_idx]
        
        # BUG: Wrong Catmull-Clark formula
        # Correct: (F + 2R + (n-3)P) / n
        # But we use: (F + R + P) / 3
        new_pos = [
            (F[0] + R[0] + P[0]) / 3.0,
            (F[1] + R[1] + P[1]) / 3.0,
            (F[2] + R[2] + P[2]) / 3.0
        ]
        
        new_positions.append(new_pos)
    
    return new_positions

def build_subdivided_mesh(vertices, faces, face_points, edge_points, new_vertex_positions):
    """Build new mesh topology. BUG: Off-by-one error in face construction."""
    all_vertices = []
    new_faces = []
    
    # Add repositioned original vertices
    all_vertices.extend(new_vertex_positions)
    original_vertex_offset = 0
    
    # Add face points
    face_point_offset = len(all_vertices)
    all_vertices.extend(face_points)
    
    # Add edge points
    edge_point_offset = len(all_vertices)
    edge_to_index = {}
    for edge, point in edge_points.items():
        edge_to_index[edge] = len(all_vertices)
        all_vertices.append(point)
    
    # Create new faces
    for face_idx, face in enumerate(faces):
        face_point_idx = face_point_offset + face_idx
        
        for i in range(len(face)):
            v1 = face[i]
            v2 = face[(i + 1) % len(face)]
            
            edge1 = tuple(sorted([face[i], face[(i - 1) % len(face)]]))
            edge2 = tuple(sorted([v1, v2]))
            
            # BUG: Wrong vertex order or missing index
            # Create quad: original_vertex, edge_point1, face_point, edge_point2
            new_face = [
                original_vertex_offset + v1,
                edge_to_index[edge2],
                face_point_idx,
                edge_to_index[edge1] + 1  # BUG: Off-by-one error here
            ]
            
            new_faces.append(new_face)
    
    return all_vertices, new_faces

def validate_topology(vertices, faces):
    """Check if all face indices reference valid vertices."""
    num_vertices = len(vertices)
    
    for face in faces:
        for v_idx in face:
            if v_idx < 0 or v_idx >= num_vertices:
                return False
    
    return True

def main():
    # Read input cube
    vertices, faces = parse_obj('data/input_cube.obj')
    
    # Build edge-face adjacency
    edge_faces = build_edge_face_map(faces)
    
    # Calculate subdivision points
    face_points = calculate_face_points(vertices, faces)
    edge_points = calculate_edge_points(vertices, faces, edge_faces, face_points)
    new_vertex_positions = reposition_original_vertices(vertices, faces, edge_faces, face_points, edge_points)
    
    # Build subdivided mesh
    subdivided_vertices, subdivided_faces = build_subdivided_mesh(
        vertices, faces, face_points, edge_points, new_vertex_positions
    )
    
    # Validate topology
    topology_valid = validate_topology(subdivided_vertices, subdivided_faces)
    
    # Output results
    result = {
        "vertex_count": len(subdivided_vertices),
        "face_count": len(subdivided_faces),
        "topology_valid": topology_valid
    }
    
    os.makedirs('/tmp', exist_ok=True)
    with open('/tmp/subdivision_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Subdivision complete:")
    print(f"  Vertices: {len(subdivided_vertices)}")
    print(f"  Faces: {len(subdivided_faces)}")
    print(f"  Topology valid: {topology_valid}")

if __name__ == '__main__':
    main()