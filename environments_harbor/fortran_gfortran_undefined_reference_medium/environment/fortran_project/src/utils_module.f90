MODULE utils_module
    IMPLICIT NONE
    
    PUBLIC :: validate_data, check_bounds
    
END MODULE utils_module

SUBROUTINE validate_data(data, n, is_valid)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: n
    REAL, DIMENSION(n), INTENT(IN) :: data
    LOGICAL, INTENT(OUT) :: is_valid
    INTEGER :: i
    
    is_valid = .TRUE.
    
    IF (n <= 0) THEN
        is_valid = .FALSE.
        RETURN
    END IF
    
    DO i = 1, n
        IF (data(i) /= data(i)) THEN
            is_valid = .FALSE.
            RETURN
        END IF
    END DO
    
END SUBROUTINE validate_data

SUBROUTINE check_bounds(data, n, min_val, max_val, within_bounds)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: n
    REAL, DIMENSION(n), INTENT(IN) :: data
    REAL, INTENT(IN) :: min_val, max_val
    LOGICAL, INTENT(OUT) :: within_bounds
    INTEGER :: i
    
    within_bounds = .TRUE.
    
    DO i = 1, n
        IF (data(i) < min_val .OR. data(i) > max_val) THEN
            within_bounds = .FALSE.
            RETURN
        END IF
    END DO
    
END SUBROUTINE check_bounds