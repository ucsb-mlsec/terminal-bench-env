public class NativeApp {
    
    static {
        try {
            System.loadLibrary("nativecalc");
            System.out.println("Native library loaded successfully!");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load native library: " + e.getMessage());
            System.err.println("java.library.path = " + System.getProperty("java.library.path"));
            throw e;
        }
    }
    
    public native int calculate(int a, int b);
    
    public static void main(String[] args) {
        try {
            NativeApp app = new NativeApp();
            
            int num1 = 42;
            int num2 = 58;
            
            System.out.println("Calling native method to calculate: " + num1 + " + " + num2);
            int result = app.calculate(num1, num2);
            
            System.out.println("Result from native method: " + result);
            System.out.println("Application completed successfully!");
            
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Error: Native library not found or native method not linked properly");
            System.err.println("Exception: " + e.getMessage());
            System.exit(1);
        } catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}