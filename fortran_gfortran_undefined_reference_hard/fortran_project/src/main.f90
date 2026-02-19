program main_program
  use module_constants
  use module_math
  use module_compute
  implicit none
  
  real(kind=8) :: result
  
  result = compute_final_value()
  
  write(*,'(F10.6)') result
  
end program main_program