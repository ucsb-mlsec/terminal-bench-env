// POV-Ray 3.7 Scene File
// Simple room scene with material property errors

#version 3.7;

global_settings {
  assumed_gamma 1.0
}

// Camera setup
camera {
  location <0, 2, -8>
  look_at <0, 1, 0>
  angle 60
}

// Main light source
light_source {
  <5, 10, -5>
  color rgb <1, 1, 1>
}

// Additional fill light
light_source {
  <-3, 5, -3>
  color rgb <0.3, 0.3, 0.4>
}

// Back wall
plane {
  z, 5
  pigment { color rgb <0.8, 0.8, 0.9> }
  finish { 
    ambient 0.1
    diffuse 0.6
    transparency 0.4
  }
}

// Left wall
plane {
  x, -6
  pigment { color rgb <0.85, 0.85, 0.95> }
  finish { 
    ambient 0.1
    diffuse 0.6
  }
}

// Right wall
plane {
  x, 6
  pigment { color rgb <0.85, 0.85, 0.95> }
  finish { 
    ambient 0.1
    diffuse 0.6
  }
}

// Floor with incorrect filter parameter
plane {
  y, 0
  pigment { 
    color rgbf <0.6, 0.5, 0.4, 0.3>
  }
  finish {
    ambient 0.15
    diffuse 0.7
    specular 0.1
  }
}

// Red box object with incorrect transmit parameter
box {
  <-2, 0, 0>, <-1, 1, 1>
  pigment { 
    color rgbt <0.9, 0.2, 0.2, 0.5>
  }
  finish {
    ambient 0.2
    diffuse 0.7
    phong 0.5
  }
}

// Green sphere object with incorrect transparency parameter
sphere {
  <1, 0.8, 1>, 0.8
  pigment { color rgb <0.2, 0.8, 0.3> }
  finish {
    ambient 0.2
    diffuse 0.6
    transparency 0.2
    phong 0.8
  }
}

// Blue cylinder object with incorrect filter parameter
cylinder {
  <2, 0, -1>, <2, 2, -1>, 0.4
  pigment { 
    color rgbf <0.2, 0.3, 0.9, 0.6>
  }
  finish {
    ambient 0.2
    diffuse 0.6
    specular 0.4
  }
}

// Additional decorative sphere (no errors)
sphere {
  <-1.5, 0.4, 2>, 0.4
  pigment { color rgb <0.9, 0.7, 0.2> }
  finish {
    ambient 0.2
    diffuse 0.7
    phong 1.0
    reflection 0.3
  }
}

// Ceiling
plane {
  y, 4
  pigment { color rgb <0.95, 0.95, 1.0> }
  finish {
    ambient 0.2
    diffuse 0.5
  }
}