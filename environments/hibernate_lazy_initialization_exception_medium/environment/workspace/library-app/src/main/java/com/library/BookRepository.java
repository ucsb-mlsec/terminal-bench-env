package com.library;

import org.hibernate.Session;
import org.hibernate.Transaction;
import org.hibernate.query.Query;
import java.util.List;

public class BookRepository {
    
    public List<Book> getAllBooks() {
        Session session = null;
        Transaction transaction = null;
        List<Book> books = null;
        
        try {
            session = HibernateUtil.getSessionFactory().openSession();
            transaction = session.beginTransaction();
            
            // HQL query to get all books
            Query<Book> query = session.createQuery("FROM Book", Book.class);
            books = query.list();
            
            transaction.commit();
        } catch (Exception e) {
            if (transaction != null) {
                transaction.rollback();
            }
            e.printStackTrace();
            throw new RuntimeException("Error retrieving books: " + e.getMessage());
        } finally {
            if (session != null) {
                session.close();
            }
        }
        
        return books;
    }
}