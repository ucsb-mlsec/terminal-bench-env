program statcalc
    use stats_module
    use utils_module
    implicit none
    
    real, dimension(12) :: data
    real :: mean_val, stddev_val
    logical :: is_valid
    
    ! Initialize sample data
    data = [23.5, 45.2, 12.8, 67.3, 34.1, 56.9, 28.4, 41.7, 19.6, 52.3, 38.8, 44.2]
    
    ! Validate the data
    is_valid = validate_data(data, size(data))
    
    if (.not. is_valid) then
        print *, 'Error: Data validation failed'
        stop 1
    end if
    
    ! Calculate statistical measures
    mean_val = calculate_mean(data, size(data))
    stddev_val = calculate_stddev(data, size(data))
    
    ! Display results
    print *, 'Statistical Analysis Results:'
    print *, '=============================='
    print *, 'Number of data points:', size(data)
    print *, 'Mean value:', mean_val
    print *, 'Standard deviation:', stddev_val
    print *, 'Minimum value:', minval(data)
    print *, 'Maximum value:', maxval(data)
    print *, ''
    print *, 'Calculation completed successfully.'
    
end program statcalc