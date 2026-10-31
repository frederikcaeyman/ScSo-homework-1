#ifndef INPUT_OUTPUT_HPP
#define INPUT_OUTPUT_HPP

#include <fstream>
#include <iostream>
#include <string>
#include <twsmatrix.hpp>

// Skip whitespace and comment lines before reading the next number.
inline void skip_comments(std::ifstream& input)
{
    input >> std::ws;
    while (input.peek() == '#') {
        std::string comment;
        std::getline(input, comment);
        input >> std::ws;
    }
}

template <typename T>
inline void read_value(std::ifstream& input, T& value)
{
    skip_comments(input);
    input >> value;
}

inline void read_vector(std::ifstream& input, tws::vector<>& values)
{
    for (int i = 0; i < values.size(); ++i) {
        read_value(input, values[i]);
    }
}

// The file gives one sample per row. Store each sample as a matrix column
// so its values are consecutive in the column-major TWS matrix storage.
inline void read_samples(std::ifstream& input, tws::matrix<>& samples)
{
    for (int k = 0; k < samples.num_columns(); ++k) {
        for (int i = 0; i < samples.num_rows(); ++i) {
            read_value(input, samples(i, k));
        }
    }
}

inline void print_vector(const tws::vector<>& values)
{
    for (int i = 0; i < values.size(); ++i) {
        if (i > 0) {
            std::cout << ' ';
        }
        std::cout << values[i];
    }
    std::cout << '\n';
}

#endif  // INPUT_OUTPUT_HPP
