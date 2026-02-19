module stats_module
    implicit none

end module stats_module

function calculate_mean(data, n) result(mean)
    implicit none
    integer, intent(in) :: n
    real, dimension(n), intent(in) :: data
    real :: mean
    
    mean = sum(data) / real(n)
end function calculate_mean

function calculate_stddev(data, n) result(stddev)
    implicit none
    integer, intent(in) :: n
    real, dimension(n), intent(in) :: data
    real :: stddev
    real :: mean
    integer :: i
    
    mean = sum(data) / real(n)
    stddev = 0.0
    do i = 1, n
        stddev = stddev + (data(i) - mean)**2
    end do
    stddev = sqrt(stddev / real(n))
end function calculate_stddev

function calculate_variance(data, n) result(variance)
    implicit none
    integer, intent(in) :: n
    real, dimension(n), intent(in) :: data
    real :: variance
    real :: mean
    integer :: i
    
    mean = sum(data) / real(n)
    variance = 0.0
    do i = 1, n
        variance = variance + (data(i) - mean)**2
    end do
    variance = variance / real(n)
end function calculate_variance