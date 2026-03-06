import java.io.*;
import java.util.*;
import java.nio.file.*;

public class SATSolver {
    
    // Static collections that accumulate data across multiple file processing
    // This is the memory leak - data is never cleared between files
    private static List<List<Integer>> allHistoricalClauses = new ArrayList<>();
    private static Map<String, List<Assignment>> solutionHistory = new HashMap<>();
    private static Set<String> processedVariables = new HashSet<>();
    
    private List<List<Integer>> clauses;
    private int numVariables;
    private int numClauses;
    private Map<Integer, Boolean> assignment;
    
    private static class Assignment {
        int variable;
        boolean value;
        List<List<Integer>> clauseSnapshot;
        
        Assignment(int variable, boolean value, List<List<Integer>> clauses) {
            this.variable = variable;
            this.value = value;
            // Store a deep copy of all clauses at this point - memory leak
            this.clauseSnapshot = new ArrayList<>();
            for (List<Integer> clause : clauses) {
                this.clauseSnapshot.add(new ArrayList<>(clause));
            }
        }
    }
    
    public SATSolver() {
        this.clauses = new ArrayList<>();
        this.assignment = new HashMap<>();
    }
    
    public boolean solve(String filename) throws IOException {
        parseCNF(filename);
        
        // Add all clauses to static historical collection - memory leak
        for (List<Integer> clause : clauses) {
            allHistoricalClauses.add(new ArrayList<>(clause));
        }
        
        // Store all variable names in static set - memory leak
        for (int i = 1; i <= numVariables; i++) {
            processedVariables.add(filename + "_var_" + i);
        }
        
        boolean result = dpll(new ArrayList<>(clauses), new HashMap<>());
        
        return result;
    }
    
    private void parseCNF(String filename) throws IOException {
        clauses.clear();
        assignment.clear();
        
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;
        
        while ((line = reader.readLine()) != null) {
            line = line.trim();
            
            if (line.isEmpty() || line.startsWith("c")) {
                continue;
            }
            
            if (line.startsWith("p cnf")) {
                String[] parts = line.split("\\s+");
                numVariables = Integer.parseInt(parts[2]);
                numClauses = Integer.parseInt(parts[3]);
                continue;
            }
            
            List<Integer> clause = new ArrayList<>();
            String[] literals = line.split("\\s+");
            
            for (String lit : literals) {
                int literal = Integer.parseInt(lit);
                if (literal == 0) break;
                clause.add(literal);
            }
            
            if (!clause.isEmpty()) {
                clauses.add(clause);
            }
        }
        
        reader.close();
    }
    
    private boolean dpll(List<List<Integer>> currentClauses, Map<Integer, Boolean> currentAssignment) {
        // Store assignment history - memory leak
        List<Assignment> fileAssignments = solutionHistory.getOrDefault(
            "current", new ArrayList<>());
        
        if (currentClauses.isEmpty()) {
            return true;
        }
        
        if (hasEmptyClause(currentClauses)) {
            return false;
        }
        
        Integer unitClause = findUnitClause(currentClauses);
        if (unitClause != null) {
            int var = Math.abs(unitClause);
            boolean value = unitClause > 0;
            
            Map<Integer, Boolean> newAssignment = new HashMap<>(currentAssignment);
            newAssignment.put(var, value);
            
            // Store this assignment with full clause snapshot - memory leak
            fileAssignments.add(new Assignment(var, value, currentClauses));
            solutionHistory.put("current", fileAssignments);
            
            List<List<Integer>> simplified = simplify(currentClauses, var, value);
            return dpll(simplified, newAssignment);
        }
        
        Integer pureLiteral = findPureLiteral(currentClauses);
        if (pureLiteral != null) {
            int var = Math.abs(pureLiteral);
            boolean value = pureLiteral > 0;
            
            Map<Integer, Boolean> newAssignment = new HashMap<>(currentAssignment);
            newAssignment.put(var, value);
            
            // Store this assignment - memory leak
            fileAssignments.add(new Assignment(var, value, currentClauses));
            solutionHistory.put("current", fileAssignments);
            
            List<List<Integer>> simplified = simplify(currentClauses, var, value);
            return dpll(simplified, newAssignment);
        }
        
        int var = selectVariable(currentClauses);
        
        Map<Integer, Boolean> newAssignment1 = new HashMap<>(currentAssignment);
        newAssignment1.put(var, true);
        List<List<Integer>> simplified1 = simplify(currentClauses, var, true);
        
        if (dpll(simplified1, newAssignment1)) {
            return true;
        }
        
        Map<Integer, Boolean> newAssignment2 = new HashMap<>(currentAssignment);
        newAssignment2.put(var, false);
        List<List<Integer>> simplified2 = simplify(currentClauses, var, false);
        
        return dpll(simplified2, newAssignment2);
    }
    
    private boolean hasEmptyClause(List<List<Integer>> clauses) {
        for (List<Integer> clause : clauses) {
            if (clause.isEmpty()) {
                return true;
            }
        }
        return false;
    }
    
    private Integer findUnitClause(List<List<Integer>> clauses) {
        for (List<Integer> clause : clauses) {
            if (clause.size() == 1) {
                return clause.get(0);
            }
        }
        return null;
    }
    
    private Integer findPureLiteral(List<List<Integer>> clauses) {
        Map<Integer, Integer> literalCount = new HashMap<>();
        
        for (List<Integer> clause : clauses) {
            for (int literal : clause) {
                literalCount.put(literal, literalCount.getOrDefault(literal, 0) + 1);
            }
        }
        
        for (int literal : literalCount.keySet()) {
            if (!literalCount.containsKey(-literal)) {
                return literal;
            }
        }
        
        return null;
    }
    
    private int selectVariable(List<List<Integer>> clauses) {
        for (List<Integer> clause : clauses) {
            if (!clause.isEmpty()) {
                return Math.abs(clause.get(0));
            }
        }
        return 1;
    }
    
    private List<List<Integer>> simplify(List<List<Integer>> clauses, int var, boolean value) {
        List<List<Integer>> result = new ArrayList<>();
        int literal = value ? var : -var;
        
        for (List<Integer> clause : clauses) {
            if (clause.contains(literal)) {
                continue;
            }
            
            List<Integer> newClause = new ArrayList<>();
            for (int lit : clause) {
                if (lit != -literal) {
                    newClause.add(lit);
                }
            }
            result.add(newClause);
        }
        
        return result;
    }
    
    public static void main(String[] args) {
        String problemDir = args.length > 0 ? args[0] : "problems/";
        
        System.out.println("SAT Solver starting...");
        System.out.println("Processing files from: " + problemDir);
        System.out.println("Heap memory limit: " + 
            (Runtime.getRuntime().maxMemory() / (1024 * 1024)) + " MB");
        
        try {
            File dir = new File(problemDir);
            File[] files = dir.listFiles((d, name) -> name.endsWith(".cnf"));
            
            if (files == null || files.length == 0) {
                System.out.println("No CNF files found in directory");
                System.exit(1);
            }
            
            Arrays.sort(files);
            int filesProcessed = 0;
            
            for (File file : files) {
                System.out.println("\nProcessing: " + file.getName());
                
                SATSolver solver = new SATSolver();
                boolean satisfiable = solver.solve(file.getPath());
                
                filesProcessed++;
                System.out.println("File: " + file.getName() + 
                    " - Result: " + (satisfiable ? "SATISFIABLE" : "UNSATISFIABLE"));
                System.out.println("Files processed so far: " + filesProcessed);
                System.out.println("Historical clauses accumulated: " + allHistoricalClauses.size());
                System.out.println("Solution history entries: " + solutionHistory.size());
                
                long usedMemory = (Runtime.getRuntime().totalMemory() - 
                    Runtime.getRuntime().freeMemory()) / (1024 * 1024);
                System.out.println("Memory used: " + usedMemory + " MB");
            }
            
            System.out.println("\n=== Processing Complete ===");
            System.out.println("Total files processed: " + filesProcessed);
            System.out.println("SUCCESS");
            
        } catch (OutOfMemoryError e) {
            System.err.println("OutOfMemoryError: Ran out of heap memory!");
            System.err.println("Historical clauses: " + allHistoricalClauses.size());
            e.printStackTrace();
            System.exit(1);
        } catch (Exception e) {
            System.err.println("Error processing files: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}