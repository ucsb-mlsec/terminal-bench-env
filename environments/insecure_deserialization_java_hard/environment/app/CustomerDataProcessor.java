import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import java.io.File;

public class CustomerDataProcessor {
    
    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java CustomerDataProcessor <xml-file-path>");
            System.exit(1);
        }
        
        String filePath = args[0];
        CustomerDataProcessor processor = new CustomerDataProcessor();
        
        try {
            processor.processCustomerData(filePath);
        } catch (Exception e) {
            System.err.println("Error processing customer data: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    public void processCustomerData(String filePath) {
        try {
            File xmlFile = new File(filePath);
            
            // Vulnerable XML parsing - no security features enabled
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            
            // Default settings allow external entity processing
            // DTD processing is enabled by default
            // Schema validation is not restricted
            // This makes the parser vulnerable to XXE attacks
            
            DocumentBuilder builder = factory.newDocumentBuilder();
            Document document = builder.parse(xmlFile);
            
            document.getDocumentElement().normalize();
            
            System.out.println("Processing XML file: " + filePath);
            System.out.println("Root element: " + document.getDocumentElement().getNodeName());
            
            // Extract customer information
            NodeList customerList = document.getElementsByTagName("customer");
            
            if (customerList.getLength() == 0) {
                System.out.println("No customer elements found in XML");
            } else {
                System.out.println("\nFound " + customerList.getLength() + " customer(s):");
                
                for (int i = 0; i < customerList.getLength(); i++) {
                    Element customer = (Element) customerList.item(i);
                    
                    String id = getElementValue(customer, "id");
                    String name = getElementValue(customer, "name");
                    String email = getElementValue(customer, "email");
                    
                    System.out.println("\nCustomer " + (i + 1) + ":");
                    System.out.println("  ID: " + id);
                    System.out.println("  Name: " + name);
                    System.out.println("  Email: " + email);
                }
            }
            
            System.out.println("\nProcessing completed successfully.");
            
        } catch (Exception e) {
            // Catches exceptions but doesn't prevent XXE attacks
            System.err.println("Exception occurred during processing: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private String getElementValue(Element parent, String tagName) {
        NodeList nodeList = parent.getElementsByTagName(tagName);
        if (nodeList.getLength() > 0) {
            Element element = (Element) nodeList.item(0);
            if (element.getFirstChild() != null) {
                return element.getFirstChild().getNodeValue();
            }
        }
        return "";
    }
}