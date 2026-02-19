#include <stdio.h>
#include <stdlib.h>

/* Internal helper functions - should not be exported */
int internal_validate_coords(int x, int y) {
    if (x < 0 || y < 0 || x > 1920 || y > 1080) {
        return 0;
    }
    return 1;
}

void* internal_allocate_buffer(int size) {
    void* buffer = malloc(size);
    if (buffer == NULL) {
        fprintf(stderr, "Failed to allocate buffer of size %d\n", size);
        return NULL;
    }
    return buffer;
}

void internal_cleanup(void) {
    printf("Performing internal cleanup operations\n");
}

/* API Version 1.0 functions */
int init_graphics_v1(void) {
    printf("Initializing graphics system v1.0\n");
    void* buffer = internal_allocate_buffer(1024);
    if (buffer == NULL) {
        return -1;
    }
    free(buffer);
    return 0;
}

void draw_line_v1(int x1, int y1, int x2, int y2) {
    if (!internal_validate_coords(x1, y1) || !internal_validate_coords(x2, y2)) {
        printf("Invalid coordinates for line\n");
        return;
    }
    printf("Drawing line v1 from (%d,%d) to (%d,%d)\n", x1, y1, x2, y2);
}

void draw_circle_v1(int x, int y, int radius) {
    if (!internal_validate_coords(x, y)) {
        printf("Invalid coordinates for circle\n");
        return;
    }
    printf("Drawing circle v1 at (%d,%d) with radius %d\n", x, y, radius);
}

void set_color_v1(int r, int g, int b) {
    printf("Setting color v1 to RGB(%d,%d,%d)\n", r, g, b);
}

void clear_screen_v1(void) {
    printf("Clearing screen v1\n");
    internal_cleanup();
}

/* API Version 2.0 functions */
void draw_rectangle_v2(int x, int y, int width, int height) {
    if (!internal_validate_coords(x, y)) {
        printf("Invalid coordinates for rectangle\n");
        return;
    }
    printf("Drawing rectangle v2 at (%d,%d) with size %dx%d\n", x, y, width, height);
}

void draw_polygon_v2(int* points, int count) {
    if (points == NULL || count < 3) {
        printf("Invalid polygon data\n");
        return;
    }
    printf("Drawing polygon v2 with %d vertices\n", count);
}

void set_color_rgba_v2(int r, int g, int b, int a) {
    printf("Setting color v2 to RGBA(%d,%d,%d,%d)\n", r, g, b, a);
}

void fill_shape_v2(int shape_id) {
    printf("Filling shape v2 with id %d\n", shape_id);
}

/* API Version 3.0 functions */
void draw_bezier_v3(float* control_points, int count) {
    if (control_points == NULL || count < 4) {
        printf("Invalid bezier curve data\n");
        return;
    }
    printf("Drawing bezier curve v3 with %d control points\n", count);
}

void apply_transform_v3(float* matrix) {
    if (matrix == NULL) {
        printf("Invalid transformation matrix\n");
        return;
    }
    printf("Applying transformation v3\n");
}

void render_scene_v3(void) {
    printf("Rendering complete scene v3\n");
    internal_cleanup();
}