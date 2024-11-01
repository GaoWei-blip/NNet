#pragma once
#ifndef LAYER_H
#define LAYER_H

#include "Neuron.h"

class Layer {
public:
    int num_neurons;             // Number of neurons in the layer
    vector<Neuron> neurons;      // Vector of neurons

    Layer();
    Layer(int num_neurons);

    ~Layer();
};
#endif
