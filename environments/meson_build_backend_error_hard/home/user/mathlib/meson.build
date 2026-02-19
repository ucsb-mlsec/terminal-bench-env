project('mathlib', 'c', version: '1.0.0')

gsl_dep = dependency('gsl')

src_files = files('src/mathlib.c')

mathlib = shared_library('mathlib',
  src_files,
  dependencies: gsl_dep,
  install: true
)