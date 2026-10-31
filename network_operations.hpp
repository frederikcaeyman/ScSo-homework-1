#ifndef NETWORK_OPERATIONS_HPP
#define NETWORK_OPERATIONS_HPP

#include <cassert>
#include <cmath>

#include "single_network_layer.hpp"

// Accept either an owning vector or a sample view without copying its values.
template <typename InputVector>
void forward_pass(NetworkLayer& layer, const InputVector& x, tws::vector<>& y)
{
    assert(x.size() == layer.n_input());
    assert(y.size() == layer.n_output());

    tws::multiply(layer.W(), x, y);
    y += layer.b();
    for (int i = 0; i < y.size(); ++i) {
        y[i] = 1.0 / (1.0 + std::exp(-y[i]));
    }
}

// Replace predictions with the common factor used by both data gradients.
template <typename TargetVector>
void compute_delta(tws::vector<>& y, const TargetVector& target)
{
    assert(y.size() == target.size());
    for (int i = 0; i < y.size(); ++i) {
        y[i] = (y[i] - target[i]) * y[i] * (1.0 - y[i]);
    }
}

#endif  // NETWORK_OPERATIONS_HPP
