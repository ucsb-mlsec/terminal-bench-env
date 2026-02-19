#version 3.7;

camera {
  location <0, 2, -5>
  look_at <0, 0, 0>
  rotate <0, 45*(2+3, 0>
}

light_source {
  <10, 20, -10>
  color rgb <1, 1, 1>
  reflexion 0.8
}

sphere {
  <0, 1, 0>, 1.5
  texture {
    pigment { color rgb <0.2, 0.5, 0.9> }
    finish { phong -0.5 ambient 0.2 diffuse 0.8 }
  }
}

plane {
  y, 0
  texture {
    pigment { checker color rgb <1, 1, 1> color rgb <0.3, 0.3, 0.3> }
    }
  }
}

sphere {
  <3, 0.8, 2>, 0.8
  texture {
    pigment { color rgbf <0.9, 0.9, 1.0, 0.95> }
    finish { 
      phong 1.0
      reflection 0.1
    }
  }
  interior { ior 0.5 }
}