#include <fstream>
#include <iostream>

#include "input_output.hpp"
#include "network_operations.hpp"

int main()
{
    std::ifstream input("input1.txt");
    if (!input) {
        std::cerr << "Could not open input1.txt\n";
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
    tws::vector<> coefficients(compute_ncoefficients_layer(n_input, n_output));
    tws::vector<> x(n_input);
    read_vector(input, coefficients);
    read_vector(input, x);
    if (!input) {
        std::cerr << "Expected all network coefficients and input values.\n";
        return 1;
    }

    // The layer views this vector, so coefficients must outlive the layer.
    // Its first entries already contain W in column-major order, followed by b.
    NetworkLayer layer(n_input, n_output, coefficients);
    tws::vector<> y(n_output);
    forward_pass(layer, x, y);

    print_vector(y);
    return 0;
}
