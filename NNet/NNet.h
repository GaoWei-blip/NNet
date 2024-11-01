#pragma once
#ifndef NNET_H
#define NNET_H

#include "Layer.h"

class NNet{
public:

	int num_layers;                                   // the number of layers in the neural network
	vector<Layer> layers;                             // the vector of layers in the neural network
	vector<int> num_neurons;                          // the number of neurons in each layer
	
	NNet(int num_layers, vector<int> num_neurons);
	~NNet() {};

	void initialize_weights();

};
#endif