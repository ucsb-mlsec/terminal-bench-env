#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <vector>
#include <string>

class NeuralNet {
private:
    int numLayers;
    int inputSize;
    int outputSize;
    float* weights;
    float* biases;

public:
    NeuralNet(const std::vector<int>& layerSizes);
    
    ~NeuralNet();
    
    std::vector<float> forward(const std::vector<float>& input);
    
    void backward(const std::vector<float>& gradients);
    
    void train(const std::vector<std::vector<float>>& inputs,
               const std::vector<std::vector<float>>& targets,
               int epochs, float learningRate);
    
    void loadModel(const std::string& path);
    
    void saveModel(const std::string& path);
};

#endif