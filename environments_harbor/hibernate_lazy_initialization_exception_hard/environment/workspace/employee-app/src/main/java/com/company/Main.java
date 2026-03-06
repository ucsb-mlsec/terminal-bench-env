package com.company;

import com.company.entity.Employee;
import com.company.util.HibernateUtil;
import org.hibernate.Session;
import org.hibernate.query.Query;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        Session session = null;
        List<Employee> employees = null;
        
        try {
            // Open a Hibernate session
            session = HibernateUtil.getSessionFactory().openSession();
            
            // Query for all Employee entities
            Query<Employee> query = session.createQuery("FROM Employee", Employee.class);
            employees = query.list();
            
            // Close the session immediately after getting the list
            session.close();
            
            // Iterate through employees and print their information
            System.out.println("Displaying Employee Information:");
            System.out.println("================================");
            
            for (Employee employee : employees) {
                // This will cause LazyInitializationException because session is closed
                // and department is lazy-loaded
                System.out.println("Employee: " + employee.getName() + 
                                 ", Email: " + employee.getEmail() + 
                                 ", Department: " + employee.getDepartment().getName());
            }
            
            System.out.println("================================");
            System.out.println("Application completed successfully!");
            
        } catch (Exception e) {
            System.err.println("An error occurred: " + e.getMessage());
            e.printStackTrace();
            throw e;
        } finally {
            // Shutdown HibernateUtil
            HibernateUtil.shutdown();
        }
    }
}