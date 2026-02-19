subroutine compute_sum(input_array, n, result)
    implicit none
    integer, intent(in) :: n
    real(8), intent(in) :: input_array(n)
    real(8), intent(out) :: result
    integer :: i
    
    result = 0.0d0
    do i = 1, n
        result = result + input_array(i)
    end do
    
end subroutine compute_sum

subroutine compute_mean(input_array, n, result)
    implicit none
    integer, intent(in) :: n
    real(8), intent(in) :: input_array(n)
    real(8), intent(out) :: result
    integer :: i
    real(8) :: sum_val
    
    sum_val = 0.0d0
    do i = 1, n
        sum_val = sum_val + input_array(i)
    end do
    result = sum_val / dble(n)
    
end subroutine compute_mean

subroutine array_multiply(input_array, n, factor, output_array)
    implicit none
    integer, intent(in) :: n
    real(8), intent(in) :: input_array(n)
    real(8), intent(in) :: factor
    real(8), intent(out) :: output_array(n)
    integer :: i
    
    do i = 1, n
        output_array(i) = input_array(i) * factor
    end do
    
end subroutine array_multiply