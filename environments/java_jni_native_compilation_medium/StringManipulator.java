public class StringManipulator {
    static {
        System.loadLibrary("StringManipulator");
    }
    
    public native String reverseString(String input);
    
    public static void main(String[] args) {
        StringManipulator manipulator = new StringManipulator();
        
        String testString = "Hello, JNI World!";
        System.out.println("Original string: " + testString);
        
        String reversed = manipulator.reverseString(testString);
        System.out.println("Reversed string: " + reversed);
    }
}