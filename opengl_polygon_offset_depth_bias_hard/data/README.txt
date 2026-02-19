DEPTH FIGHTING TEST RESULTS - DATA FORMAT DOCUMENTATION

OVERVIEW
========
This dataset contains measurements from 50 different rendering scenarios designed to evaluate depth fighting (z-fighting) artifacts under various conditions. Each test scenario represents a unique combination of camera configuration, depth buffer precision, polygon orientation, and polygon offset parameters. The data was collected to identify optimal polygon offset parameters that eliminate depth fighting artifacts across all tested configurations.

COLUMN DESCRIPTIONS
===================

scenario_id
  Unique integer identifier for each test scenario (1-50)

near_plane
  Camera near clipping plane distance in world units. Range: 0.1 to 10.0
  Represents the closest distance at which geometry is rendered

far_plane
  Camera far clipping plane distance in world units. Range: 100 to 10000
  Represents the farthest distance at which geometry is rendered

polygon_angle
  Angle in degrees between the polygon surface normal and the view direction
  Range: 0° to 85°
  0° = polygon facing directly toward camera (perpendicular to view)
  90° = polygon edge-on to camera (parallel to view direction)
  Higher angles are more susceptible to depth fighting artifacts

depth_bits
  Depth buffer precision in bits: 16, 24, or 32
  Higher precision provides better depth resolution but may not be available on all hardware

offset_factor_tested
  The polygon offset factor parameter value tested in this scenario
  This value scales with the polygon's slope in screen space

offset_units_tested
  The polygon offset units parameter value tested in this scenario
  This value represents a constant offset in depth buffer units

depth_fighting_severity
  Quantified metric measuring the severity of depth fighting artifacts
  Range: 0.0 to 1.0
  0.0 = no visible artifacts, perfect depth separation
  1.0 = severe z-fighting with significant visual problems
  Values below 0.1 are considered acceptable for production use

artifacts_visible
  Boolean flag (TRUE/FALSE) indicating whether visual artifacts were observed
  Generally correlates with depth_fighting_severity >= 0.1

INTERPRETATION NOTES
====================
A successful polygon offset solution must achieve depth_fighting_severity < 0.1 across ALL 50 test scenarios. The challenge is that different scenarios have competing requirements - some need larger offsets due to steep angles or low precision, while others need smaller offsets to avoid incorrect depth ordering.

The relationship between these parameters and depth fighting is non-linear and depends on the mathematical formulas documented in depth_precision_specs.txt. Analysts should refer to that document for the precise calculations used to determine how offset_factor and offset_units affect the final depth buffer values.