import java.io.*;

public class VulnerableApp {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Usage: java VulnerableApp <serialized-file-path>");
            System.exit(1);
        }

        String filename = args[0];
        System.out.println("Reading serialized object from: " + filename);

        FileInputStream fileIn = null;
        ObjectInputStream objectIn = null;

        try {
            fileIn = new FileInputStream(filename);
            objectIn = new ObjectInputStream(fileIn);
            
            // VULNERABLE: Deserializing user-controlled input without validation
            Object obj = objectIn.readObject();
            
            System.out.println("Successfully deserialized object: " + obj.getClass().getName());
            System.out.println("Object contents: " + obj.toString());
            
        } catch (FileNotFoundException e) {
            System.err.println("Error: File not found - " + filename);
            e.printStackTrace();
        } catch (IOException e) {
            System.err.println("Error: IO exception during deserialization");
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            System.err.println("Error: Class not found during deserialization");
            e.printStackTrace();
        } finally {
            try {
                if (objectIn != null) {
                    objectIn.close();
                }
                if (fileIn != null) {
                    fileIn.close();
                }
            } catch (IOException e) {
                System.err.println("Error closing streams");
                e.printStackTrace();
            }
        }
    }
}