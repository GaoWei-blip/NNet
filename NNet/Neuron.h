#pragma once
#ifndef NEURON_H
#define NEURON_H

#include <vector>
using namespace std;

class Neuron {
public:
    // forward
    float actv;                            // current layer activation output 
    vector<float> out_weights;             // current layer weight
    float bias;
    float z;                               // current layer output z = sum(out_weights¡Áactv)+bias

    // derivative 
    float dactv;
    vector<float> dw;
    float dbias;
    float dz;

    Neuron();
    Neuron(int num_out_weights);
    ~Neuron();
};

#endif
