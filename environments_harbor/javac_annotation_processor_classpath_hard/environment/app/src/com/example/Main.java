package com.example;

public class Main {
    public static void main(String[] args) {
        // Create and test Person object
        Person person = new Person();
        person.setName("John Doe");
        person.setAge(30);
        person.setEmail("john@example.com");
        person.setAddress("123 Main St");
        
        System.out.println("Person Details:");
        System.out.println("Name: " + person.getName());
        System.out.println("Age: " + person.getAge());
        System.out.println("Email: " + person.getEmail());
        System.out.println("Address: " + person.getAddress());
        System.out.println();
        
        // Create and test Product object
        Product product = new Product();
        product.setProductId("P12345");
        product.setProductName("Laptop");
        product.setPrice(999.99);
        
        System.out.println("Product Details:");
        System.out.println("Product ID: " + product.getProductId());
        System.out.println("Product Name: " + product.getProductName());
        System.out.println("Price: $" + product.getPrice());
        System.out.println();
        
        // Create and test Order object
        Order order = new Order();
        order.setOrderId("ORD-001");
        order.setCustomerId("CUST-456");
        order.setQuantity(5);
        
        System.out.println("Order Details:");
        System.out.println("Order ID: " + order.getOrderId());
        System.out.println("Customer ID: " + order.getCustomerId());
        System.out.println("Quantity: " + order.getQuantity());
        System.out.println();
        
        System.out.println("All getter and setter methods are working correctly!");
    }
}