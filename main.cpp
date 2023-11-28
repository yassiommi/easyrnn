//
//  main.cpp
//  simplified-rnn
//
//

#include <iostream>
#include <vector>
#include <cmath>
#include <random>

// Define a simple Matrix structure for basic matrix operations
struct Matrix {
    int rows;
    int cols;
    std::vector<std::vector<double>> data;

    Matrix(int rows, int cols) : rows(rows), cols(cols) {
        data.resize(rows, std::vector<double>(cols, 0.0));
    }

    void randomize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                data[i][j] = dis(gen);
            }
        }
    }
};

// Define activation functions (tanh and softmax)
double tanhActivation(double x) {
    return std::tanh(x);
}

std::vector<double> softmax(const std::vector<double>& x) {
    std::vector<double> result;
    double sum = 0.0;

    for (double val : x) {
        sum += std::exp(val);
    }

    for (double val : x) {
        result.push_back(std::exp(val) / sum);
    }

    return result;
}

class SimpleRNN {
private:
    int inputSize;
    int hiddenSize;
    int outputSize;

    Matrix Wxh; // Input to hidden weights
    Matrix Whh; // Hidden to hidden weights
    Matrix Why; // Hidden to output weights
    std::vector<double> bh; // Hidden bias
    std::vector<double> by; // Output bias

    std::vector<double> hPrev; // Previous hidden state

    double learningRate;

public:
    SimpleRNN(int inputSize, int hiddenSize, int outputSize, double learningRate)
        : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize), learningRate(learningRate),
          Wxh(hiddenSize, inputSize), Whh(hiddenSize, hiddenSize), Why(outputSize, hiddenSize) {
        Wxh.randomize();
        Whh.randomize();
        Why.randomize();

        bh.resize(hiddenSize, 0.0);
        by.resize(outputSize, 0.0);

        hPrev.resize(hiddenSize, 0.0);
    }

    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> h(hiddenSize, 0.0);

        for (int i = 0; i < hiddenSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                h[i] += Wxh.data[i][j] * input[j];
            }
            h[i] += bh[i];
            h[i] = tanhActivation(h[i] + Whh.data[i][i] * hPrev[i]); // Recurrent connection
        }

        std::vector<double> output(outputSize, 0.0);

        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < hiddenSize; ++j) {
                output[i] += Why.data[i][j] * h[j];
            }
            output[i] += by[i];
        }

        // Store the current hidden state for the next iteration
        hPrev = h;

        return softmax(output);
    }

    void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                std::vector<double> prediction = forward(inputs[i]);

                std::vector<double> lossGradient;
                for (size_t j = 0; j < prediction.size(); ++j) {
                    lossGradient.push_back(prediction[j] - targets[i][j]);
                }

                // Backpropagation through time (BPTT)
                std::vector<double> dh(hiddenSize, 0.0);
                for (int j = 0; j < outputSize; ++j) {
                    for (int k = 0; k < hiddenSize; ++k) {
                        Why.data[j][k] -= learningRate * lossGradient[j] * hPrev[k];
                        dh[k] += Why.data[j][k] * lossGradient[j];
                    }
                    by[j] -= learningRate * lossGradient[j];
                }

                for (int j = 0; j < hiddenSize; ++j) {
                    dh[j] *= (1 - hPrev[j] * hPrev[j]); // Derivative of tanh
                    for (int k = 0; k < inputSize; ++k) {
                        Wxh.data[j][k] -= learningRate * dh[j] * inputs[i][k];
                    }
                    for (int k = 0; k < hiddenSize; ++k) {
                        Whh.data[j][k] -= learningRate * dh[j] * hPrev[k];
                    }
                    bh[j] -= learningRate * dh[j];
                }
            }
        }
    }
};

int main() {
    int inputSize = 3;
    int hiddenSize = 4;
    int outputSize = 2;
    double learningRate = 0.1;

    // Example input and target sequences
    std::vector<std::vector<double>> inputs = {
        {0.1, 0.2, 0.3},
        {0.4, 0.5, 0.6},
        {0.7, 0.8, 0.9}
    };

    std::vector<std::vector<double>> targets = {
        {0.2, 0.3},
        {0.5, 0.6},
        {0.8, 0.9}
    };

    SimpleRNN rnn(inputSize, hiddenSize, outputSize, learningRate);
    rnn.train(inputs, targets, 100);

    for (const auto& input : inputs) {
        std::vector<double> prediction = rnn.forward(input);
        std::cout << "Input: ";
        for (double val : input) {
            std::cout << val << " ";
        }
        std::cout << "\nPrediction: ";
        for (double val : prediction) {
            std::cout << val << " ";
        }
        std::cout << "\n\n";
    }

    return 0;
}
