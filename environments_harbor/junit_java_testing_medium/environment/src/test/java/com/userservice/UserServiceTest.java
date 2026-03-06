package com.userservice;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class UserServiceTest {
    
    private UserService userService;
    
    @BeforeEach
    public void setUp() {
        userService = new UserService();
    }
    
    @Test
    public void testCreateUser() {
        User user = userService.createUser("1", "John", "john@test.com", 30);
        assertNotNull(user);
        assertEquals("1", user.getId());
        assertEquals("John", user.getName());
        assertEquals("john@test.com", user.getEmail());
        assertEquals(30, user.getAge());
    }
    
    @Test
    public void testGetUser() {
        userService.createUser("1", "John", "john@test.com", 30);
        User user = userService.getUser("1");
        assertNotNull(user);
        assertEquals("1", user.getId());
    }
    
    @Test
    public void testUpdateEmail() {
        userService.createUser("1", "John", "john@test.com", 30);
        userService.updateEmail("1", "newemail@test.com");
        User user = userService.getUser("1");
        assertEquals("newemail@test.com", user.getEmail());
    }
    
    @Test
    public void testGetUserCount() {
        userService.createUser("1", "John", "john@test.com", 30);
        userService.createUser("2", "Jane", "jane@test.com", 25);
        userService.createUser("3", "Bob", "bob@test.com", 35);
        assertEquals(3, userService.getUserCount());
    }
    
    @Test
    public void testDeleteUser() {
        userService.createUser("1", "John", "john@test.com", 30);
        boolean deleted = userService.deleteUser("1");
        assertTrue(deleted);
        assertNull(userService.getUser("1"));
    }
}