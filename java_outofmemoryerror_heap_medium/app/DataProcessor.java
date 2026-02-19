import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;

public class DataProcessor {
    private List<DataRecord> allRecords;
    private Map<String, List<DataRecord>> groupedData;
    private List<String> processedResults;
    
    public DataProcessor() {
        allRecords = new ArrayList<>();
        groupedData = new HashMap<>();
        processedResults = new ArrayList<>();
    }
    
    static class DataRecord {
        String id;
        String category;
        String name;
        double value;
        String timestamp;
        String[] additionalFields;
        
        public DataRecord(String[] fields) {
            if (fields.length >= 5) {
                this.id = fields[0];
                this.category = fields[1];
                this.name = fields[2];
                this.value = parseDouble(fields[3]);
                this.timestamp = fields[4];
                this.additionalFields = new String[fields.length];
                System.arraycopy(fields, 0, additionalFields, 0, fields.length);
            }
        }
        
        private double parseDouble(String s) {
            try {
                return Double.parseDouble(s);
            } catch (NumberFormatException e) {
                return 0.0;
            }
        }
        
        @Override
        public String toString() {
            return String.join(",", additionalFields);
        }
    }
    
    public void loadDataFiles(String directoryPath) throws IOException {
        System.out.println("Loading data files from: " + directoryPath);
        File dataDir = new File(directoryPath);
        
        if (!dataDir.exists() || !dataDir.isDirectory()) {
            System.err.println("Data directory not found: " + directoryPath);
            return;
        }
        
        File[] files = dataDir.listFiles((dir, name) -> name.endsWith(".csv"));
        if (files == null || files.length == 0) {
            System.out.println("No CSV files found in directory");
            return;
        }
        
        for (File file : files) {
            System.out.println("Processing file: " + file.getName());
            loadFile(file);
        }
        
        System.out.println("Total records loaded: " + allRecords.size());
    }
    
    private void loadFile(File file) throws IOException {
        List<String> lines = new ArrayList<>();
        
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            boolean isHeader = true;
            
            while ((line = reader.readLine()) != null) {
                if (isHeader) {
                    isHeader = false;
                    continue;
                }
                lines.add(line);
            }
        }
        
        // Store all lines in memory before processing
        for (String line : lines) {
            String[] fields = line.split(",");
            DataRecord record = new DataRecord(fields);
            allRecords.add(record);
            
            // Create duplicates for memory intensive processing
            for (int i = 0; i < 3; i++) {
                allRecords.add(new DataRecord(fields));
            }
        }
    }
    
    public void processData() {
        System.out.println("Processing data records...");
        
        // Group data by category - memory intensive operation
        for (DataRecord record : allRecords) {
            String category = record.category != null ? record.category : "UNKNOWN";
            
            if (!groupedData.containsKey(category)) {
                groupedData.put(category, new ArrayList<>());
            }
            groupedData.get(category).add(record);
        }
        
        System.out.println("Grouped into " + groupedData.size() + " categories");
        
        // Perform aggregations - keeping all intermediate results in memory
        for (Map.Entry<String, List<DataRecord>> entry : groupedData.entrySet()) {
            String category = entry.getKey();
            List<DataRecord> records = entry.getValue();
            
            double sum = 0.0;
            double max = Double.MIN_VALUE;
            double min = Double.MAX_VALUE;
            
            for (DataRecord record : records) {
                sum += record.value;
                max = Math.max(max, record.value);
                min = Math.min(min, record.value);
            }
            
            double average = records.size() > 0 ? sum / records.size() : 0.0;
            
            String result = String.format("%s,COUNT=%d,SUM=%.2f,AVG=%.2f,MIN=%.2f,MAX=%.2f",
                category, records.size(), sum, average, min, max);
            processedResults.add(result);
            
            // Store transformed records
            for (DataRecord record : records) {
                String transformed = String.format("PROCESSED_%s_%s_%.2f",
                    category, record.id, record.value * 1.5);
                processedResults.add(transformed);
            }
        }
        
        System.out.println("Generated " + processedResults.size() + " processed results");
    }
    
    public void writeOutput(String outputPath) throws IOException {
        System.out.println("Writing output to: " + outputPath);
        
        File outputFile = new File(outputPath);
        outputFile.getParentFile().mkdirs();
        
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile))) {
            writer.write("CATEGORY,STATISTICS\n");
            
            for (String result : processedResults) {
                writer.write(result);
                writer.newLine();
            }
        }
        
        System.out.println("Output written successfully");
    }
    
    public void cleanup() {
        if (allRecords != null) allRecords.clear();
        if (groupedData != null) groupedData.clear();
        if (processedResults != null) processedResults.clear();
    }
    
    public static void main(String[] args) {
        System.out.println("=== DataProcessor Starting ===");
        DataProcessor processor = new DataProcessor();
        
        try {
            String dataPath = "../data/";
            String outputPath = "../data/processed_output.csv";
            
            processor.loadDataFiles(dataPath);
            processor.processData();
            processor.writeOutput(outputPath);
            
            System.out.println("=== DataProcessor Completed Successfully ===");
            
        } catch (OutOfMemoryError e) {
            System.err.println("ERROR: Out of Memory!");
            System.err.println("The application requires more heap memory to process the dataset.");
            System.err.println("Please increase the heap size using -Xmx JVM option.");
            e.printStackTrace();
            System.exit(1);
            
        } catch (IOException e) {
            System.err.println("ERROR: IO Exception occurred");
            e.printStackTrace();
            System.exit(1);
            
        } catch (Exception e) {
            System.err.println("ERROR: Unexpected exception");
            e.printStackTrace();
            System.exit(1);
            
        } finally {
            processor.cleanup();
        }
    }
}