#ifndef SINGLE_NETWORK_LAYER_HPP
#define SINGLE_NETWORK_LAYER_HPP

#include <random>
#include <twsmatrix.hpp>

/**
 * @brief Computes the number of coefficients (weights and biases)
 *        for a network layer with n_input inputs and n_output outputs.
 */
int compute_ncoefficients_layer(int n_input, int n_output)
{
    return n_input * n_output + n_output;
}

/**
 * @brief A simple neural network layer class.
 */
class NetworkLayer {
   public:
    NetworkLayer(int n_in, int n_out, tws::vectorview<> coefficients)
        : n_in(n_in), n_out(n_out), coefficients(coefficients)
    {
        assert(coefficients.size() == compute_ncoefficients_layer(n_in, n_out));
    }

    /**
     * @brief Return the number of input features
     */
    int n_input() const { return n_in; }

    /**
     * @brief Return the number of output features
     */
    int n_output() const { return n_out; }

    /**
     * @brief Access the weight matrix W (n_output x n_input)
     */
    tws::matrixview<> W()
    {
        return tws::matrixview<>(n_out, n_in, coefficients.data(), n_out);
    }

    /**
     * @brief Access the bias vector b (n_output)
     */
    tws::vectorview<> b()
    {
        return tws::vectorview<>(n_out, coefficients.data() + n_in * n_out);
    }

    /**
     * @brief Access the full coefficients vector (W and b concatenated)
     */
    tws::vectorview<> coeffs()
    {
        return tws::vectorview<>(coefficients.size(), coefficients.data());
    }

    /**
     * @brief Set all coefficients (weights and biases) to zero
     */
    void set_to_zero()
    {
        for (int i = 0; i < coefficients.size(); ++i) {
            coefficients[i] = 0.0;
        }
    }

    /**
     * @brief Set all coefficients (weights and biases) to random values
     *       in the range [0, 0.05]
     */
    void set_to_random()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 0.05);
        for (int i = 0; i < coefficients.size(); ++i) {
            coefficients[i] = dis(gen);
        }
    }

   private:
    const int n_in;
    const int n_out;
    // This vector contains the coefficients of the neural network
    // The coefficients are stored in the following order:
    // W (n_output x n_input) stored in column-major order
    // b (n_output)
    tws::vectorview<> coefficients;
};

#endif  // SINGLE_NETWORK_LAYER_HPP