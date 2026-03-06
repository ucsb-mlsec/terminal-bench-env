#version 3.7;

global_settings { assumed_gamma 1.0 }

// Camera definition
camera {
  location <0, 5, -10>
  look_at <0, 0, 0>
  angle 45
}

// Light sources
light_source {
  <10, 20, -15>
  color rgb <1, 1, 1>
}

light_source {
  <-5, 10, -8>
  color rgb <0.5, 0.5, 0.6>
}

// Ground plane
plane {
  y, 0
  pigment { color rgb <0.2, 0.5, 0.2> }
  finish { ambient 0.1 diffuse 0.6 }
}

// First sphere with error - negative radius
sphere {
  <0, -2.5, 0>, -2.5
  pigment { color rgb <1, 0, 0> }
  finish { ambient 0.2 diffuse 0.7 }
}

// Second sphere with vector error
sphere {
  <2, 1, 3, 4>, 1.0
  pigment { color rgb <0, 0, 1> }
  finish { ambient 0.3 diffuse 0.6 specular 0.5 }
}

// Box object missing opening brace
box
  <-1, 0, -1>, <1, 2, 1>
  pigment { color rgb <0.8, 0.8, 0> }
  finish { ambient 2.3 diffuse 0.5 }
}

// Another sphere for scene depth
sphere {
  <-3, 1.5, 2>, 1.2
  pigment { color rgb <1, 0.5, 0> }
  finish { ambient 0.15 diffuse 0.8 }
}

// Cylinder with unclosed declaration
cylinder {
  <0, 0, -2>, <0, 3, -2>, 0.5
  pigment { color rgb <0.5, 0, 0.5> }
  finish { ambient 0.2 diffuse 0.7 }

// Additional sphere with wrong vector separator
sphere {
  <4; 1; -1>, 0.8
  pigment { color rgb <0, 1, 1> }
  finish { ambient 0.2 diffuse 0.6 specular 0.3 }
}