package com.example.auth;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class AuthService {

    @Autowired
    private UserService userService;

    @Autowired
    private PermissionService permissionService;

    public boolean authenticate(String username, String password) {
        boolean userExists = userService.findUser(username);
        return userExists && password != null && password.length() > 0;
    }

    public boolean validateToken(String token) {
        return token != null && token.startsWith("TOKEN_") && permissionService.hasPermission(token);
    }

    public void logout(String username) {
        userService.updateUserStatus(username, "LOGGED_OUT");
        permissionService.revokePermissions(username);
    }
}