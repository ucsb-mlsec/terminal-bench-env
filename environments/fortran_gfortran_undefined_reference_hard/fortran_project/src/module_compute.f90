module module_compute
  use module_constants
  use module_math
  implicit none

contains

  function compute_result() result(res)
    real(kind=8) :: res
    real(kind=8) :: value1, value2
    
    ! Use constants from module_constants
    value1 = PI_VALUE * 2.0d0
    value2 = E_VALUE
    
    ! Use function from module_math
    res = calculate_sum(value1, value2)
    
    ! Apply additional computation with constant
    res = res * SCALE_FACTOR
    
  end function compute_result

end module module_compute