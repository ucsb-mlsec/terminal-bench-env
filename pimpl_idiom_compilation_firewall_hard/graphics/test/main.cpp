#include "Canvas.h"
#include "Palette.h"
#include "Pipeline.h"

int main() {
    // Test Canvas
    Canvas canvas(800, 600);
    canvas.clear();
    canvas.setPixel(10, 20, 0xFF0000);
    canvas.getWidth();
    canvas.getHeight();
    
    // Test Palette
    Palette palette;
    palette.addColor("red", 0xFF0000);
    palette.addColor("green", 0x00FF00);
    palette.getColor("red");
    palette.getColorCount();
    
    // Test Pipeline
    Pipeline pipeline;
    pipeline.addStage("stage1");
    pipeline.addStage("stage2");
    pipeline.execute();
    pipeline.getStageCount();
    
    return 0;
}