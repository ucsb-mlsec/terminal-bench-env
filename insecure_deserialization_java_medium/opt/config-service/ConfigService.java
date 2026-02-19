import java.io.*;

/**
 * Configuration Service for Web Application
 * Manages persistent storage of application settings
 * WARNING: This implementation has known security vulnerabilities
 */
public class ConfigService {
    
    private static final String CONFIG_FILE = "/var/lib/config-service/app.dat";
    
    /**
     * AppConfig - Serializable configuration object
     * Contains all application settings including credentials
     */
    static class AppConfig implements Serializable {
        private static final long serialVersionUID = 1L;
        
        String dbHost;
        String dbUser;
        String dbPassword;
        String apiEndpoint;
        boolean featureToggleX;
        int maxConnections;
        
        public AppConfig(String dbHost, String dbUser, String dbPassword, 
                        String apiEndpoint, boolean featureToggleX, int maxConnections) {
            this.dbHost = dbHost;
            this.dbUser = dbUser;
            this.dbPassword = dbPassword;
            this.apiEndpoint = apiEndpoint;
            this.featureToggleX = featureToggleX;
            this.maxConnections = maxConnections;
        }
        
        @Override
        public String toString() {
            return "AppConfig{" +
                    "dbHost='" + dbHost + '\'' +
                    ", dbUser='" + dbUser + '\'' +
                    ", apiEndpoint='" + apiEndpoint + '\'' +
                    ", featureToggleX=" + featureToggleX +
                    ", maxConnections=" + maxConnections +
                    '}';
        }
    }
    
    /**
     * Saves configuration to disk using Java serialization
     */
    public void saveConfig(AppConfig config) throws IOException {
        File configFile = new File(CONFIG_FILE);
        configFile.getParentFile().mkdirs();
        
        try (FileOutputStream fos = new FileOutputStream(configFile);
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            oos.writeObject(config);
            System.out.println("Configuration saved successfully to " + CONFIG_FILE);
        }
    }
    
    /**
     * Loads configuration from disk using Java deserialization
     * VULNERABLE: Uses ObjectInputStream.readObject() which can execute arbitrary code
     * An attacker who can modify app.dat can achieve remote code execution
     */
    public AppConfig loadConfig() throws IOException, ClassNotFoundException {
        File configFile = new File(CONFIG_FILE);
        
        if (!configFile.exists()) {
            System.out.println("Config file not found, returning default configuration");
            return new AppConfig("localhost", "admin", "default", 
                               "http://api.example.com", false, 10);
        }
        
        try (FileInputStream fis = new FileInputStream(configFile);
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            // SECURITY VULNERABILITY: Deserializing untrusted data
            AppConfig config = (AppConfig) ois.readObject();
            System.out.println("Configuration loaded successfully from " + CONFIG_FILE);
            return config;
        }
    }
    
    /**
     * Demo main method showing save/load operations
     */
    public static void main(String[] args) {
        ConfigService service = new ConfigService();
        
        try {
            // Create sample configuration
            AppConfig config = new AppConfig(
                "db.production.com",
                "app_user",
                "secure_password_123",
                "https://api.production.com/v1",
                true,
                50
            );
            
            // Save configuration
            System.out.println("Saving configuration...");
            service.saveConfig(config);
            
            // Load configuration
            System.out.println("\nLoading configuration...");
            AppConfig loadedConfig = service.loadConfig();
            System.out.println("Loaded: " + loadedConfig);
            
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Error during configuration operation: " + e.getMessage());
            e.printStackTrace();
        }
    }
}