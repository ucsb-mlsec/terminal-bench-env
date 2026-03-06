package com.example.app;

import com.example.annotations.AutoGetter;

public class Person {
    @AutoGetter
    private String name;
    
    @AutoGetter
    private int age;
    
    @AutoGetter
    private String email;
    
    @AutoGetter
    private String address;
    
    public Person(String name, int age, String email, String address) {
        this.name = name;
        this.age = age;
        this.email = email;
        this.address = address;
    }
    
    public static void main(String[] args) {
        Person person = new Person("John Doe", 30, "john.doe@example.com", "123 Main St");
        
        System.out.println("Person Details:");
        System.out.println("Name: " + person.getName());
        System.out.println("Age: " + person.getAge());
        System.out.println("Email: " + person.getEmail());
        System.out.println("Address: " + person.getAddress());
        
        System.out.println("\nAnnotation processor successfully generated getter methods!");
    }
}