#include <cassert>
#include <fstream>
#include <iostream>

#include "input_output.hpp"
#include "network_operations.hpp"

void train(NetworkLayer& layer, const tws::matrix<>& inputs,
           const tws::matrix<>& targets, double gamma, double alpha,
           int n_epochs)
{
    assert(inputs.num_rows() == layer.n_input());
    assert(targets.num_rows() == layer.n_output());
    assert(inputs.num_columns() == targets.num_columns());
    const int n_samples = inputs.num_columns();
    assert(n_samples > 0);

    // Allocate temporary vectors once, before the training loops.
    auto coefficients = layer.coeffs();
    tws::vector<> gradient(coefficients.size());
    NetworkLayer gradient_layer(layer.n_input(), layer.n_output(), gradient);
    auto gradient_W = gradient_layer.W();
    auto gradient_b = gradient_layer.b();
    tws::vector<> y(layer.n_output());

    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        gradient_layer.set_to_zero();

        for (int k = 0; k < n_samples; ++k) {
            auto x = inputs.column(k);
            auto target = targets.column(k);
            forward_pass(layer, x, y);

            compute_delta(y, target);

            for (int j = 0; j < layer.n_input(); ++j) {
                for (int i = 0; i < layer.n_output(); ++i) {
                    gradient_W(i, j) += y[i] * x[j];
                }
            }
            gradient_b += y;
        }

        // Average the data gradient and add regularization once per epoch.
        // Update only after processing every sample (batch gradient descent).
        for (int i = 0; i < coefficients.size(); ++i) {
            coefficients[i] -= alpha * (gradient[i] / n_samples
                                        + gamma * coefficients[i]);
        }
    }
}

int main()
{
    std::ifstream input("input3.txt");
    if (!input) {
        std::cerr << "Could not open input3.txt\n";
        return 1;
    }

    int n_input = 0;
    int n_output = 0;
    int n_samples = 0;
    double gamma = 0.0;
    double alpha = 0.0;
    int n_epochs = 0;
    read_value(input, n_input);
    read_value(input, n_output);
    read_value(input, n_samples);
    read_value(input, gamma);
    read_value(input, alpha);
    read_value(input, n_epochs);

    if (!input || n_input <= 0 || n_output <= 0 || n_samples <= 0
        || gamma < 0.0 || alpha <= 0.0 || n_epochs < 0) {
        std::cerr << "Expected positive dimensions, sample count, and learning rate,\n"
                  << "and nonnegative gamma and epoch count.\n";
        return 1;
    }

    tws::matrix<> inputs(n_input, n_samples);
    tws::matrix<> targets(n_output, n_samples);
    read_samples(input, inputs);
    read_samples(input, targets);
    if (!input) {
        std::cerr << "Expected all training input and output values.\n";
        return 1;
    }

    tws::vector<> coefficients(compute_ncoefficients_layer(n_input, n_output));
    NetworkLayer layer(n_input, n_output, coefficients);
    layer.set_to_random();
    train(layer, inputs, targets, gamma, alpha, n_epochs);

    tws::vector<> y(n_output);
    for (int k = 0; k < n_samples; ++k) {
        forward_pass(layer, inputs.column(k), y);
        print_vector(y);
    }
    return 0;
}
