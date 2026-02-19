subroutine calc_mean(data, n, mean_out)
    implicit none
    integer, intent(in) :: n
    real(kind=8), intent(in) :: data(n)
    real(kind=8), intent(out) :: mean_out
    integer :: i
    real(kind=8) :: sum_val
    
    sum_val = 0.0d0
    do i = 1, n
        sum_val = sum_val + data(i)
    end do
    mean_out = sum_val / real(n, kind=8)
    
end subroutine calc_mean

subroutine calc_stddev(data, n, stddev_out)
    implicit none
    integer, intent(in) :: n
    real(kind=8), intent(in) :: data(n)
    real(kind=8), intent(out) :: stddev_out
    integer :: i
    real(kind=8) :: mean_val, sum_sq, variance
    
    ! Calculate mean
    mean_val = 0.0d0
    do i = 1, n
        mean_val = mean_val + data(i)
    end do
    mean_val = mean_val / real(n, kind=8)
    
    ! Calculate sum of squared differences
    sum_sq = 0.0d0
    do i = 1, n
        sum_sq = sum_sq + (data(i) - mean_val)**2
    end do
    
    ! Calculate sample standard deviation (n-1 denominator)
    variance = sum_sq / real(n - 1, kind=8)
    stddev_out = sqrt(variance)
    
end subroutine calc_stddev