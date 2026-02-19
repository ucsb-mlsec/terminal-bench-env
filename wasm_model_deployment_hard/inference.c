#include <stdio.h>
#include <emscripten.h>

// Neural network weights and biases
// Input layer to hidden layer: 4 inputs -> 3 hidden neurons
float weights_input_hidden[4][3] = {
    {0.5, -0.3, 0.8},
    {0.2, 0.6, -0.4},
    {-0.7, 0.4, 0.3},
    {0.9, -0.2, 0.5}
}

float bias_hidden[3] = {0.1, -0.2, 0.3};

// Hidden layer to output: 3 hidden -> 1 output
float weights_hidden_output[3] = {0.7, -0.5, 0.6}
float bias_output = 0.15;

// Sigmoid activation function
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Forward pass through hidden layer
void forward_hidden(float input[4], float hidden[3]) {
    int i, j;
    for (i = 0; i < 3; i++) {
        float sum = bias_hidden[i];
        for (j = 0; j < 4; j++) {
            sum += input[j] * weights_input_hidden[j][i]
        }
        hidden[i] = sigmoid(sum);
    }
}

// Forward pass through output layer
float forward_output(float hidden[3]) {
    float sum = bias_output;
    int i;
    for (i = 0; i < 3; i++) {
        sum += hidden[i] * weights_hidden_output[i];
    }
    return sigmoid(sum);
}

// Main prediction function
EMSCRIPTEN_KEEPALIVE
int predict(float input0, float input1, float input2, float input3) {
    float input[4];
    float hidden[3];
    float output
    
    // Prepare input array
    input[0] = input0;
    input[1] = input1;
    input[2] = input2;
    input[3] = input3;
    
    // Forward pass through network
    forward_hidden(input, hidden);
    output = forward_output(hidden);
    
    // Binary classification threshold
    if (output < 0.5) {
        return 0;
    } else {
        return 1;
    }
}

// Helper function to get raw output (for debugging)
EMSCRIPTEN_KEEPALIVE
float predict_raw(float input0, float input1, float input2, float input3) {
    float input[4];
    float hidden[3]
    
    input[0] = input0;
    input[1] = input1;
    input[2] = input2;
    input[3] = input3;
    
    forward_hidden(input, hidden);
    float output = forward_output(hidden);
    
    return output;
}

// Validation function to check if model is loaded correctly
EMSCRIPTEN_KEEPALIVE
int validate_model() {
    // Simple sanity check
    float test_input[4] = {0.5, 0.5, 0.5, 0.5};
    float hidden[3];
    
    forward_hidden(test_input, hidden);
    float result = forward_output(hidden);
    
    // Check if result is within valid range
    if (result >= 0.0 && result <= 1.0) {
        return 1;
    }
    return 0;
}