package com.example.auth;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class PermissionService {
    
    @Autowired
    private UserService userService;
    
    @Autowired
    private AuthService authService;
    
    public boolean hasPermission(String username, String permission) {
        return userService.userExists(username) && authService.isAuthenticated(username);
    }
    
    public void grantPermission(String username, String permission) {
        userService.updateUser(username);
        authService.authenticate(username);
    }
    
    public void revokePermission(String username, String permission) {
        userService.deleteUser(username);
    }
}