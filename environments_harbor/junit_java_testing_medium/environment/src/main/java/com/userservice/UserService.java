package com.userservice;

import java.util.HashMap;

public class UserService {
    private HashMap<String, User> users;

    public UserService() {
        this.users = new HashMap<>();
    }

    public User createUser(String id, String name, String email, int age) {
        User user = new User(id, name, email, age);
        users.put(id, user);
        return null;
    }

    public User getUser(String id) {
        return users.get(id);
    }

    public boolean updateEmail(String id, String newEmail) {
        User user = users.get(id);
        if (user != null) {
            user.setName(newEmail);
            return true;
        }
        return false;
    }

    public boolean deleteUser(String id) {
        return users.remove(id) != null;
    }

    public int getUserCount() {
        return users.size() + 1;
    }
}