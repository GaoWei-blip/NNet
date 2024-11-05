#include "Trainer.h"

Trainer::Trainer(NNet net, Dataset dataset):net(net), dataset(dataset), alpha(0.01f), epoch(3) {
	cost.resize(net.num_neurons[net.num_layers - 1]);
}

Trainer::~Trainer(){}

void Trainer::train_neural_net()
{
	// Gradient Descent
	for (int it = 0; it < epoch; it++)
	{
		float epoch_cost = 0;
		for (int i = 0; i < static_cast<int>(dataset.trainX.size()); i++)
		{
			feed_input(i);
			forward_prop();
			epoch_cost += compute_cost(i);
			printf("Epoch [%i][%i] Cost: %f\n", it + 1, i, epoch_cost);
			back_prop(i);
		}
		//printf("Epoch [%i] Cost: %f\n", it, epoch_cost / static_cast<int>(dataset.trainX.size()));
	}
}

void Trainer::feed_input(int i)
{
	for (int j = 0; j < net.num_neurons[0]; j++)
	{
		net.layers[0].neurons[j].actv = dataset.trainX[i][j];
	}
}

float Trainer::sigmoid(float x) {
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

float Trainer::compute_cost(int i)
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

	//printf("data [%i] Cost: %f\n", i, tcost);
	return tcost;
}

void Trainer::back_prop(int p)
{
	vector<int> num_neurons = net.num_neurons;
	int num_layers = net.num_layers;

	// Output Layer
	for (int j = 0; j < num_neurons[num_layers - 1]; j++)
	{
		net.layers[num_layers - 1].neurons[j].dz = net.layers[num_layers - 1].neurons[j].actv - dataset.trainY[p][j];

		for (int k = 0; k < num_neurons[num_layers - 2]; k++)
		{
			net.layers[num_layers - 2].neurons[k].dw[j] = net.layers[num_layers - 1].neurons[j].dz * net.layers[num_layers - 2].neurons[k].actv;
		}

		net.layers[num_layers - 1].neurons[j].dbias = net.layers[num_layers - 1].neurons[j].dz;
	}

	// Hidden Layers
	for (int i = num_layers - 2; i > 0; i--)
	{
		for (int j = 0; j < num_neurons[i]; j++)
		{
			net.layers[i].neurons[j].dz = 0;
			for (int k = 0; k < num_neurons[i + 1]; k++) { 
				net.layers[i].neurons[j].dz += net.layers[i].neurons[j].dw[k] * net.layers[i + 1].neurons[k].dz * 
					net.layers[i].neurons[j].z * (1 - net.layers[i].neurons[j].z);
			}

			for (int k = 0; k < num_neurons[i - 1]; k++)
			{
				net.layers[i - 1].neurons[k].dw[j] = net.layers[i].neurons[j].dz * net.layers[i - 1].neurons[k].actv;
			}

			net.layers[i].neurons[j].dbias = net.layers[i].neurons[j].dz;
		}
	}

	// update weights and biases

}