#pragma once
#ifndef NEURON_H
#define NEURON_H

#include <vector>
using namespace std;

class Neuron {
public:
    // forward
    vector<float> w;                       // current layer input weights
    float bias;
    float z;                               // current layer output z = sum(w¡Ápre_layer_actv)+bias
    float actv;                            // current layer activation output

    // derivative 
    vector<float> dw;
    float dbias;
    float dz;

    Neuron();
    Neuron(int num_input_weights);
    ~Neuron();
};

#endif
