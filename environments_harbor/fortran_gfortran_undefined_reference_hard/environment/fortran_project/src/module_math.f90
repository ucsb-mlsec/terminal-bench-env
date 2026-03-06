module module_math
    use module_constants
    implicit none
    
contains
    
    function compute_circle_area(radius) result(area)
        real(kind=8), intent(in) :: radius
        real(kind=8) :: area
        
        area = PI * radius * radius
    end function compute_circle_area
    
    function compute_sphere_volume(radius) result(volume)
        real(kind=8), intent(in) :: radius
        real(kind=8) :: volume
        
        volume = (4.0d0 / 3.0d0) * PI * radius * radius * radius
    end function compute_sphere_volume
    
    subroutine apply_gravity_force(mass, force)
        real(kind=8), intent(in) :: mass
        real(kind=8), intent(out) :: force
        
        force = mass * GRAVITY
    end subroutine apply_gravity_force
    
end module module_math