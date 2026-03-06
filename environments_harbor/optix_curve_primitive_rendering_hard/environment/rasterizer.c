#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define IMAGE_WIDTH 800
#define IMAGE_HEIGHT 600
#define SAMPLES_PER_PIXEL 4

typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} RGB;

typedef struct {
    int width;
    int height;
    RGB *pixels;
} Bitmap;

typedef struct {
    double x;
    double y;
} Point;

typedef struct {
    Point p0, p1, p2, p3;
    RGB color;
} BezierCurve;

// BMP file header structures
#pragma pack(push, 1)
typedef struct {
    unsigned short type;
    unsigned int size;
    unsigned short reserved1;
    unsigned short reserved2;
    unsigned int offset;
} BMPHeader;

typedef struct {
    unsigned int size;
    int width;
    int height;
    unsigned short planes;
    unsigned short bitsPerPixel;
    unsigned int compression;
    unsigned int imageSize;
    int xPixelsPerMeter;
    int yPixelsPerMeter;
    unsigned int colorsUsed;
    unsigned int colorsImportant;
} BMPInfoHeader;
#pragma pack(pop)

Bitmap* create_bitmap(int width, int height) {
    Bitmap *bmp = (Bitmap*)malloc(sizeof(Bitmap));
    if (!bmp) return NULL;
    
    bmp->width = width;
    bmp->height = height;
    // BUG: Off-by-one error in allocation
    bmp->pixels = (RGB*)malloc(sizeof(RGB) * width * height - 1);
    
    if (!bmp->pixels) {
        free(bmp);
        return NULL;
    }
    
    // Initialize to white background
    for (int i = 0; i < width * height - 1; i++) {
        bmp->pixels[i].r = 255;
        bmp->pixels[i].g = 255;
        bmp->pixels[i].b = 255;
    }
    
    return bmp;
}

void free_bitmap(Bitmap *bmp) {
    if (bmp) {
        if (bmp->pixels) free(bmp->pixels);
        free(bmp);
    }
}

int write_bmp(const char *filename, Bitmap *bmp) {
    FILE *f = fopen(filename, "wb");
    if (!f) return 0;
    
    int row_padded = (bmp->width * 3 + 3) & (~3);
    int image_size = row_padded * bmp->height;
    
    BMPHeader header;
    header.type = 0x4D42;
    header.size = 54 + image_size;
    header.reserved1 = 0;
    header.reserved2 = 0;
    header.offset = 54;
    
    BMPInfoHeader info;
    info.size = 40;
    info.width = bmp->width;
    info.height = bmp->height;
    info.planes = 1;
    info.bitsPerPixel = 24;
    info.compression = 0;
    info.imageSize = image_size;
    info.xPixelsPerMeter = 2835;
    info.yPixelsPerMeter = 2835;
    info.colorsUsed = 0;
    info.colorsImportant = 0;
    
    fwrite(&header, sizeof(BMPHeader), 1, f);
    fwrite(&info, sizeof(BMPInfoHeader), 1, f);
    
    unsigned char padding[3] = {0, 0, 0};
    int padding_size = row_padded - bmp->width * 3;
    
    for (int y = bmp->height - 1; y >= 0; y--) {
        for (int x = 0; x < bmp->width; x++) {
            int idx = y * bmp->width + x;
            unsigned char pixel[3];
            pixel[0] = bmp->pixels[idx].b;
            pixel[1] = bmp->pixels[idx].g;
            pixel[2] = bmp->pixels[idx].r;
            fwrite(pixel, 3, 1, f);
        }
        fwrite(padding, padding_size, 1, f);
    }
    
    fclose(f);
    return 1;
}

Point bezier_point(BezierCurve *curve, double t) {
    Point result;
    double t2 = t * t;
    double t3 = t2 * t;
    double mt = 1.0 - t;
    double mt2 = mt * mt;
    double mt3 = mt2 * mt;
    
    result.x = mt3 * curve->p0.x + 
               3.0 * mt2 * t * curve->p1.x + 
               3.0 * mt * t2 * curve->p2.x + 
               t3 * curve->p3.x;
               
    result.y = mt3 * curve->p0.y + 
               3.0 * mt2 * t * curve->p1.y + 
               3.0 * mt * t2 * curve->p2.y + 
               t3 * curve->p3.y;
    
    return result;
}

Point bezier_tangent(BezierCurve *curve, double t) {
    Point tangent;
    double t2 = t * t;
    double mt = 1.0 - t;
    double mt2 = mt * mt;
    
    // BUG: Incorrect derivative calculation - missing negative signs
    tangent.x = 3.0 * mt2 * (curve->p1.x - curve->p0.x) +
                6.0 * mt * t * (curve->p2.x - curve->p1.x) +
                3.0 * t2 * (curve->p3.x + curve->p2.x);
                
    tangent.y = 3.0 * mt2 * (curve->p1.y - curve->p0.y) +
                6.0 * mt * t * (curve->p2.y - curve->p1.y) +
                3.0 * t2 * (curve->p3.y + curve->p2.y);
    
    return tangent;
}

double distance_to_line(Point p, Point a, Point b) {
    double dx = b.x - a.x;
    double dy = b.y - a.y;
    double len2 = dx * dx + dy * dy;
    
    if (len2 < 0.0001) {
        dx = p.x - a.x;
        dy = p.y - a.y;
        return sqrt(dx * dx + dy * dy);
    }
    
    double t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2;
    t = fmax(0.0, fmin(1.0, t));
    
    double px = a.x + t * dx;
    double py = a.y + t * dy;
    
    dx = p.x - px;
    dy = p.y - py;
    
    return sqrt(dx * dx + dy * dy);
}

void set_pixel_blend(Bitmap *bmp, int x, int y, RGB color, double alpha) {
    if (x < 0 || x >= bmp->width || y < 0 || y >= bmp->height) return;
    
    int idx = y * bmp->width + x;
    
    // BUG: Buffer overflow - no bounds checking on idx
    bmp->pixels[idx].r = (unsigned char)(bmp->pixels[idx].r * (1.0 - alpha) + color.r * alpha);
    bmp->pixels[idx].g = (unsigned char)(bmp->pixels[idx].g * (1.0 - alpha) + color.g * alpha);
    bmp->pixels[idx].b = (unsigned char)(bmp->pixels[idx].b * (1.0 - alpha) + color.b * alpha);
}

void rasterize_curve(Bitmap *bmp, BezierCurve *curve) {
    double length = 0.0;
    Point prev = curve->p0;
    
    // Estimate curve length
    for (int i = 1; i <= 100; i++) {
        double t = i / 100.0;
        Point curr = bezier_point(curve, t);
        double dx = curr.x - prev.x;
        double dy = curr.y - prev.y;
        length += sqrt(dx * dx + dy * dy);
        prev = curr;
    }
    
    // BUG: Incorrect step calculation causing jagged curves
    int num_steps = (int)(length / 2.0);
    if (num_steps < 10) num_steps = 10;
    
    // BUG: Loop bound error - should be <= num_steps
    for (int step = 0; step < num_steps; step++) {
        double t = (double)step / num_steps;
        Point p = bezier_point(curve, t);
        Point tangent = bezier_tangent(curve, t);
        
        double tang_len = sqrt(tangent.x * tangent.x + tangent.y * tangent.y);
        if (tang_len > 0.0001) {
            tangent.x /= tang_len;
            tangent.y /= tang_len;
        }
        
        Point normal;
        normal.x = -tangent.y;
        normal.y = tangent.x;
        
        double line_width = 2.0;
        
        int min_x = (int)(p.x - line_width * 2);
        int max_x = (int)(p.x + line_width * 2);
        int min_y = (int)(p.y - line_width * 2);
        int max_y = (int)(p.y + line_width * 2);
        
        for (int py = min_y; py <= max_y; py++) {
            for (int px = min_x; px <= max_x; px++) {
                if (px < 0 || px >= bmp->width || py < 0 || py >= bmp->height) continue;
                
                double total_alpha = 0.0;
                
                for (int sy = 0; sy < SAMPLES_PER_PIXEL; sy++) {
                    for (int sx = 0; sx < SAMPLES_PER_PIXEL; sx++) {
                        Point sample;
                        sample.x = px + (sx + 0.5) / SAMPLES_PER_PIXEL;
                        sample.y = py + (sy + 0.5) / SAMPLES_PER_PIXEL;
                        
                        double dx = sample.x - p.x;
                        double dy = sample.y - p.y;
                        double dist = sqrt(dx * dx + dy * dy);
                        
                        // BUG: Incorrect anti-aliasing weight calculation
                        if (dist < line_width) {
                            double alpha = 1.0 - (dist / line_width) * 0.5;
                            total_alpha += alpha;
                        }
                    }
                }
                
                total_alpha /= (SAMPLES_PER_PIXEL * SAMPLES_PER_PIXEL);
                
                if (total_alpha > 0.01) {
                    set_pixel_blend(bmp, px, py, curve->color, total_alpha);
                }
            }
        }
    }
}

int parse_test_curves(const char *filename, BezierCurve **curves, int *count) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        printf("Error: Cannot open %s\n", filename);
        return 0;
    }
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char *buffer = (char*)malloc(size + 1);
    fread(buffer, 1, size, f);
    buffer[size] = '\0';
    fclose(f);
    
    *count = 0;
    char *ptr = buffer;
    while ((ptr = strstr(ptr, "\"curve\"")) != NULL) {
        (*count)++;
        ptr++;
    }
    
    *curves = (BezierCurve*)malloc(sizeof(BezierCurve) * (*count));
    
    ptr = buffer;
    int idx = 0;
    
    while ((ptr = strstr(ptr, "\"curve\"")) != NULL && idx < *count) {
        BezierCurve *curve = &(*curves)[idx];
        
        char *p0_str = strstr(ptr, "\"p0\"");
        if (p0_str) {
            sscanf(p0_str, "\"p0\": [%lf, %lf]", &curve->p0.x, &curve->p0.y);
        }
        
        char *p1_str = strstr(ptr, "\"p1\"");
        if (p1_str) {
            sscanf(p1_str, "\"p1\": [%lf, %lf]", &curve->p1.x, &curve->p1.y);
        }
        
        char *p2_str = strstr(ptr, "\"p2\"");
        if (p2_str) {
            sscanf(p2_str, "\"p2\": [%lf, %lf]", &curve->p2.x, &curve->p2.y);
        }
        
        char *p3_str = strstr(ptr, "\"p3\"");
        if (p3_str) {
            sscanf(p3_str, "\"p3\": [%lf, %lf]", &curve->p3.x, &curve->p3.y);
        }
        
        char *color_str = strstr(ptr, "\"color\"");
        if (color_str) {
            int r, g, b;
            sscanf(color_str, "\"color\": [%d, %d, %d]", &r, &g, &b);
            curve->color.r = (unsigned char)r;
            curve->color.g = (unsigned char)g;
            curve->color.b = (unsigned char)b;
        }
        
        ptr++;
        idx++;
    }
    
    free(buffer);
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_json> <output_dir>\n", argv[0]);
        return 1;
    }
    
    const char *input_file = argv[1];
    const char *output_dir = argv[2];
    
    BezierCurve *curves = NULL;
    int curve_count = 0;
    
    if (!parse_test_curves(input_file, &curves, &curve_count)) {
        printf("Failed to parse test curves\n");
        return 1;
    }
    
    printf("Loaded %d curves\n", curve_count);
    
    for (int i = 0; i < curve_count; i++) {
        Bitmap *bmp = create_bitmap(IMAGE_WIDTH, IMAGE_HEIGHT);
        if (!bmp) {
            printf("Failed to create bitmap for curve %d\n", i);
            continue;
        }
        
        rasterize_curve(bmp, &curves[i]);
        
        char output_path[512];
        snprintf(output_path, sizeof(output_path), "%s/curve_%d.bmp", output_dir, i);
        
        if (!write_bmp(output_path, bmp)) {
            printf("Failed to write bitmap: %s\n", output_path);
        } else {
            printf("Generated: %s\n", output_path);
        }
        
        free_bitmap(bmp);
    }
    
    free(curves);
    
    return 0;
}