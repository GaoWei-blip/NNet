#include "Trainer.h"

Trainer::Trainer(NNet net, Dataset dataset):net(net), dataset(dataset), full_cost(0.0), count(0) {
	cost.resize(net.num_neurons[net.num_layers - 1]);
}

Trainer::~Trainer(){}

void Trainer::train_neural_net()
{
	// Gradient Descent
	for (int it = 0; it < 20000; it++)
	{
		for (int i = 0; i < static_cast<int>(dataset.trainX.size()); i++)
		{
			feed_input(i);
			//forward_prop();
			//compute_cost(i);
			//back_prop(i);
			//update_weights();
		}
	}
}

void Trainer::feed_input(int i)
{
	for (int j = 0; j < net.num_neurons[0]; j++)
	{
		net.layers[0].neurons[j].actv = dataset.trainX[i][j];
	}
}

float Trainer::sigmoid(int x) {
	return 1 / (1 + expf(-x));
}

void Trainer::forward_prop()
{
	for (int i = 1; i < net.num_layers; i++)
	{
		for (int j = 0; j < net.num_neurons[i]; j++)
		{
			// Computer weighted sum z
			for (int k = 0; k < net.num_neurons[i - 1]; k++)
			{
				net.layers[i].neurons[j].z += (net.layers[i - 1].neurons[k].out_weights[j]) * (net.layers[i - 1].neurons[k].actv);
			}
			net.layers[i].neurons[j].z += net.layers[i].neurons[j].bias;

			// Sigmoid Activation function
			net.layers[i].neurons[j].actv = sigmoid(net.layers[i].neurons[j].z);
			//printf("Output: %d\n", (int)round(net.layers[i].neurons[j].actv));
			
		}
	}
}

void Trainer::compute_cost(int i)
{
	float tmpcost = 0;
	float tcost = 0;
	int num_layers = net.num_layers;

	for(int j = 0; j < net.num_neurons[num_layers - 1]; j++)
	{
		// MSE loss
		tmpcost = dataset.trainY[i][j] - net.layers[num_layers - 1].neurons[j].actv;
		cost[j] = (tmpcost * tmpcost) / 2;
		tcost = tcost + cost[j];
	}

	full_cost = (full_cost + tcost) / count;
	count++;
	// printf("Full Cost: %f\n",full_cost);
}

void Trainer::back_prop(int p)
{
	vector<int> num_neurons = net.num_neurons;
	int num_layers = net.num_layers;
	vector<Layer> lay = net.layers;

	// Output Layer
	for (int j = 0; j < num_neurons[num_layers - 1]; j++)
	{
		lay[num_layers - 1].neurons[j].dz = lay[num_layers - 1].neurons[j].actv - dataset.trainY[p][j];

		for (int k = 0; k < num_neurons[num_layers - 2]; k++)
		{
			lay[num_layers - 2].neurons[k].dw[j] = lay[num_layers - 1].neurons[j].dz * lay[num_layers - 2].neurons[k].actv;
		}

		lay[num_layers - 1].neurons[j].dbias = lay[num_layers - 1].neurons[j].dz;
	}

	// Hidden Layers
	for (int i = num_layers - 2; i > 0; i--)
	{
		for (int j = 0; j < num_neurons[i]; j++)
		{
			for (int k = 0; k < num_neurons[i + 1]; k++) { 
				lay[i].neurons[j].dz += lay[i].neurons[j].dw[k] * lay[i + 1].neurons[k].dz * lay[i].neurons[j].z * (1 - lay[i].neurons[j].z);
			}
			

			/*if (lay[i].neurons[j].z >= 0)
			{
				lay[i].neurons[j].dz = lay[i].neurons[j].dactv;
			}
			else
			{
				lay[i].neurons[j].dz = 0;
			}*/

			for (int k = 0; k < num_neurons[i - 1]; k++)
			{
				lay[i - 1].neurons[k].dw[j] = lay[i].neurons[j].dz * lay[i - 1].neurons[k].actv;

				if (i > 1)
				{
					lay[i - 1].neurons[k].dactv = lay[i - 1].neurons[k].out_weights[j] * lay[i].neurons[j].dz;
				}
			}

			lay[i].neurons[j].dbias = lay[i].neurons[j].dz;
		}


	}
}