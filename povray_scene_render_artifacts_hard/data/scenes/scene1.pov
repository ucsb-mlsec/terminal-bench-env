#version 3.7;

global_settings { assumed_gamma 1.0 }

camera {
  location <1, 2>
  look_at <0, 0, 0>
  angle 45
}

light_source {
  <10, 20, -30>
  sphere {
    <0, 1, 0>, 1.5
    pigment { color rgb <0.2, 0.5, 0.8> }
    finish { 
      ambient 0.1
      diffuse 0.6
      transmit 1.5
      specular 0.8
    }
  }
}

plane {
  <0, 1, 0>, 0
  pigment { color rgb (1, 0, 0) }
  finish { reflection 0.2 }
}

camara {
  perspective
}

box {
  <-1, 0, -1>, <1, 2, 1>
  pigment { color rgb <0.9, 0.7, 0.3> }
  finish { phong 1 }
}

cylinder {
  <0, 0, 0>, <0, 3, 0>, 0.5
  pigment { color rgb <0.1, 0.9, 0.1> }
  finish { metallic }
}