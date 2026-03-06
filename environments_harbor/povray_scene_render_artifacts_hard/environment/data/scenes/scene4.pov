// POV-Ray Scene File - Geometric Shapes Demonstration
// This scene contains various geometric objects with lighting and camera setup

camera {
    location [0, 5, -10]
    look_at <0, 0, 0>
}

// Main light source
lite_source {
    <10, 15, -20>
    color rgb <1, 1, 1>
}

sphere {
    <0, 1, 0>, 1.5
    texture {
        pigment { color rgb <1 2 3> }
        finish { phong 0.9 }
    }
    rotate <0, 45, 0>
    texture {
        pigment { colour rgb <0.8, 0.2, 0.3> }
        finish { reflection 0.3 }
    }
}

box {
    <-1, 0, -1>, <1, 0.5, 1>
    pigment { color rgb <0.2, 0.6, 0.9> }
    translate <0, 1, 0,>
}

cone {
    <0, 0, 0>, 0.8
    <0, 2, 0>, 0.0
    texture {
        pigment { color rgb <0.9, 0.7, 0.1> }
        finish { ambient 0.2 diffuse 0.8 }
    }
    rotate 0, 45, 0
    translate <3, 0, 2>
}

plane {
    <0, 1, 0>, 0
    pigment {
        checker color rgb <1, 1, 1> color rgb <0.3, 0.3, 0.3>
    }
}