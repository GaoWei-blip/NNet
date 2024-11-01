#include "Neuron.h"

Neuron::Neuron(int num_out_weights)
    : actv(0), bias(0), z(0), dactv(0), dbias(0), dz(0) {
    out_weights.resize(num_out_weights, 0.0f);
    dw.resize(num_out_weights, 0.0f);
}

Neuron::~Neuron() {
    // Due to the use of std::vector, there is no need to manually release memory
}

Neuron::Neuron() {}