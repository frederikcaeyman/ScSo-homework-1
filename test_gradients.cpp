#include <cassert>
#include <fstream>
#include <iostream>

#include "input_output.hpp"
#include "network_operations.hpp"

// Reuse the temporary vector y to evaluate the loss without new vectors.
double loss(NetworkLayer& layer, const tws::vector<>& x,
            const tws::vector<>& target, double gamma, tws::vector<>& y)
{
    assert(target.size() == layer.n_output());
    forward_pass(layer, x, y);
    y -= target;

    // The coefficient vector contains both W and b, so its squared norm
    // accounts for both regularization terms.
    return 0.5 * tws::dot(y, y)
           + 0.5 * gamma * tws::dot(layer.coeffs(), layer.coeffs());
}

void finite_differences(NetworkLayer& layer, const tws::vector<>& x,
                        const tws::vector<>& target, double gamma,
                        tws::vector<>& gradient, tws::vector<>& y)
{
    auto coefficients = layer.coeffs();
    assert(gradient.size() == coefficients.size());
    const double h = 1.0e-6;

    for (int i = 0; i < coefficients.size(); ++i) {
        const double original = coefficients[i];

        coefficients[i] = original + h;
        const double loss_plus = loss(layer, x, target, gamma, y);

        coefficients[i] = original - h;
        const double loss_minus = loss(layer, x, target, gamma, y);

        // Restore the original coefficient before moving to the next one.
        coefficients[i] = original;
        gradient[i] = (loss_plus - loss_minus) / (2.0 * h);
    }
}

void backpropagation(NetworkLayer& layer, const tws::vector<>& x,
                     const tws::vector<>& target, double gamma,
                     tws::vector<>& gradient, tws::vector<>& y)
{
    assert(target.size() == layer.n_output());
    forward_pass(layer, x, y);

    compute_delta(y, target);

    // Views give the gradient the same W-then-b layout as the coefficients.
    NetworkLayer gradient_layer(layer.n_input(), layer.n_output(), gradient);
    auto gradient_W = gradient_layer.W();
    auto gradient_b = gradient_layer.b();
    auto W = layer.W();
    auto b = layer.b();

    for (int j = 0; j < layer.n_input(); ++j) {
        for (int i = 0; i < layer.n_output(); ++i) {
            gradient_W(i, j) = y[i] * x[j] + gamma * W(i, j);
        }
    }
    for (int i = 0; i < layer.n_output(); ++i) {
        gradient_b[i] = y[i] + gamma * b[i];
    }
}

int main()
{
    std::ifstream input("input2.txt");
    if (!input) {
        std::cerr << "Could not open input2.txt\n";
        return 1;
    }

    int n_input = 0;
    int n_output = 0;
    read_value(input, n_input);
    read_value(input, n_output);
    if (!input || n_input <= 0 || n_output <= 0) {
        std::cerr << "Expected positive input and output dimensions.\n";
        return 1;
    }

    const int n_coefficients = compute_ncoefficients_layer(n_input, n_output);
    tws::vector<> coefficients(n_coefficients);
    tws::vector<> x(n_input);
    tws::vector<> target(n_output);
    double gamma = 0.0;

    read_vector(input, coefficients);
    read_vector(input, x);
    read_value(input, gamma);
    read_vector(input, target);
    if (!input) {
        std::cerr << "Expected coefficients, input, gamma, and target values.\n";
        return 1;
    }

    NetworkLayer layer(n_input, n_output, coefficients);
    tws::vector<> gradient_fd(n_coefficients);
    tws::vector<> gradient_bp(n_coefficients);
    tws::vector<> y(n_output);

    finite_differences(layer, x, target, gamma, gradient_fd, y);
    backpropagation(layer, x, target, gamma, gradient_bp, y);

    // Each line contains W's gradient in column-major order, followed by b's.
    print_vector(gradient_fd);
    print_vector(gradient_bp);
    return 0;
}
