package com.example.auth;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class UserService {
    
    @Autowired
    private AuthService authService;
    
    @Autowired
    private PermissionService permissionService;
    
    public String getUserById(Long id) {
        authService.authenticate("user" + id);
        return "User with ID: " + id;
    }
    
    public String createUser(String username) {
        permissionService.checkPermission("CREATE_USER");
        return "Created user: " + username;
    }
    
    public boolean validateUser(String username) {
        return authService.isAuthenticated(username) && permissionService.hasPermission(username, "READ");
    }
}