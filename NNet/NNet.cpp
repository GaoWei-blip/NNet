#include "NNet.h"
#include <ctime>


NNet::NNet(int num_layers, vector<int> num_neurons):num_layers(num_layers), num_neurons(num_neurons)
{
	for (int i = 0; i < num_layers; i++){
		Layer layer = Layer(num_neurons[i]);
		layer.num_neurons = num_neurons[i];
		for (int j = 0; j < num_neurons[i]; j++){
			if (i < (num_layers - 1))
			{
				Neuron neuron = Neuron(num_neurons[i + 1]);
				layer.neurons.emplace_back(neuron);
			}
			else
			{
				Neuron neuron = Neuron();
				layer.neurons.emplace_back(neuron);
			}
		}
		layers.emplace_back(layer);
	}
	initialize_weights();
}

void NNet::initialize_weights() {
	// random seed
    //srand(static_cast<unsigned>(time(nullptr)));
	srand(static_cast<unsigned>(3407));

	for (int i = 0; i < num_layers - 1; i++){            // the last layer has no weight output 
		for (int j = 0; j < num_neurons[i]; j++) {
			for (int k = 0; k < num_neurons[i + 1]; k++) {
				layers[i].neurons[j].w[k] = static_cast<float>(rand()) / RAND_MAX;
				//printf("(%d)w[%d][%d]: %f\n", i + 1, k + 1, j + 1, layers[i].neurons[j].w[k]);
				layers[i].neurons[j].dw[k] = 0.0;
			}
			if (i > 0) {
				layers[i].neurons[j].bias = static_cast<float>(rand()) / RAND_MAX;
				//printf("(%d):bias[%d]: %f\n", i + 1, j + 1, layers[i].neurons[j].bias);
			}
		}
		printf("\n");
	}

	// the last layer has bias
	for (int j = 0; j < num_neurons[num_layers - 1]; j++) {
		layers[num_layers - 1].neurons[j].bias = static_cast<float>(rand()) / RAND_MAX;
		//printf("(%d)bias[%d]: %f\n", num_layers, j+1, layers[num_layers - 1].neurons[j].bias);
	}
	
	
}

