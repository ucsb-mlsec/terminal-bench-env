package com.library;

import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.Transaction;
import org.hibernate.cfg.Configuration;
import org.hibernate.query.Query;

import java.util.List;

public class Main {
    
    private static SessionFactory sessionFactory;
    
    public static void main(String[] args) {
        try {
            // Initialize Hibernate SessionFactory
            sessionFactory = new Configuration()
                    .configure("hibernate.cfg.xml")
                    .addAnnotatedClass(Book.class)
                    .addAnnotatedClass(Review.class)
                    .buildSessionFactory();
            
            // Retrieve books from database
            List<Book> books = getBooks();
            
            // Display book information with reviews
            displayBooksWithReviews(books);
            
            System.out.println("Application completed successfully!");
            
        } catch (Exception e) {
            System.err.println("Error occurred: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        } finally {
            if (sessionFactory != null) {
                sessionFactory.close();
            }
        }
    }
    
    private static List<Book> getBooks() {
        Session session = null;
        Transaction transaction = null;
        List<Book> books = null;
        
        try {
            // Open session and begin transaction
            session = sessionFactory.openSession();
            transaction = session.beginTransaction();
            
            // Query for all books using HQL
            Query<Book> query = session.createQuery("FROM Book", Book.class);
            books = query.getResultList();
            
            // Commit transaction
            transaction.commit();
            
            System.out.println("Retrieved " + books.size() + " books from database");
            
        } catch (Exception e) {
            if (transaction != null) {
                transaction.rollback();
            }
            throw new RuntimeException("Failed to retrieve books", e);
        } finally {
            // Close the session - this is critical for demonstrating the lazy init problem
            if (session != null) {
                session.close();
            }
        }
        
        return books;
    }
    
    private static void displayBooksWithReviews(List<Book> books) {
        System.out.println("\n=== Books and Reviews ===\n");
        
        for (Book book : books) {
            System.out.println("Book: " + book.getTitle());
            System.out.println("Author: " + book.getAuthor());
            System.out.println("ISBN: " + book.getIsbn());
            System.out.println("Reviews:");
            
            // This will cause LazyInitializationException because session is closed
            // and reviews are lazy-loaded by default
            List<Review> reviews = book.getReviews();
            
            if (reviews == null || reviews.isEmpty()) {
                System.out.println("  No reviews available");
            } else {
                for (Review review : reviews) {
                    System.out.println("  - Rating: " + review.getRating() + "/5");
                    System.out.println("    Review: " + review.getReviewText());
                    System.out.println("    Reviewer: " + review.getReviewerName());
                }
            }
            System.out.println();
        }
    }
}