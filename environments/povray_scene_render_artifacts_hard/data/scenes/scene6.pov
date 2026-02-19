#version 3.7;

global_settings { assumed_gamma 1.0 }

camera {
  location <0, 5, -10>
  look_at <0, 0, 0>
  pigment { color rgbf <0.8, 0.9, 1.0, 1.8> }
}

light_source {
  <10, 20, -15>
  color rgb <1, 1, 1>
}

texture {
  pigment { color rgb <0.7, 0.3, 0.2> }
  finish { 
    ambient 0.2
    diffuse 0.8
    scale <1, -2, 1>
  }
}

torus {
  2.0, 0.5
  pigmnt { color rgb <0.2, 0.6, 0.9> }
  finish {
    specular 0.6
    roughness 0.05
  }
  rotate <90, 0, 0>
  translate 1, 2, 3
}

cone {
  <0, 0, 0>, 1.5
  <0, 3, 0>, 0
  texture {
    pigment { color rgb <0.9, 0.7, 0.1> }
    finish { diffuse 1.7 }
  }
  translate <-3, 0, 2>
}

box {
  <-1, 0, -1>, <1, 0.5, 1>
  texture {
    pigment {
      checker
      color rgb <1, 1, 1>
      color rgb <0.3, 0.3, 0.3>
      scale 0.5
    }
    finish {
      ambient 0.1
      diffuse 0.7
      reflection 0.2
    }
  }
  translate <3, 0, 0>
}

sphere {
  <0, 1, 0>, 1.2
  texture {
    pigment { color rgb <0.8, 0.2, 0.4> }
    normal { bumps 0.3 scale 0.2 }
    finish {
      phong 0.9
      phong_size 60
      ambient 0.15
    }
  }
}

plane {
  y, 0
  texture {
    pigment { color rgb <0.5, 0.5, 0.5> }
    finish { ambient 0.3 diffuse 0.7 }
  }
}