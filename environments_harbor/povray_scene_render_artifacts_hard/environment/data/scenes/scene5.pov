// POV-Ray Scene Description Language File
// A simple scene with sphere and plane
// This file contains no errors and should render correctly

#version 3.7;

global_settings {
  assumed_gamma 1.0
}

// Camera definition
camera {
  location <0, 2, -5>
  look_at <0, 0, 0>
  angle 45
}

// Light source
light_source {
  <10, 10, -10>
  color rgb <1, 1, 1>
}

// Red sphere at origin
sphere {
  <0, 0, 0>, 1
  pigment {
    color rgb <1, 0, 0>
  }
  finish {
    phong 0.9
    ambient 0.2
    diffuse 0.7
  }
}

// Checkered plane below
plane {
  <0, 1, 0>, -1
  pigment {
    checker
    color rgb <1, 1, 1>
    color rgb <0, 0, 0>
    scale 0.5
  }
  finish {
    ambient 0.3
    diffuse 0.8
  }
}