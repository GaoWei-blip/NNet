#include "Trainer.h"

Trainer::Trainer(NNet& net, Dataset& dataset, int epoch, float alpha):net(net), dataset(dataset), alpha(alpha), epoch(epoch) {
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
			//printf("Epoch [%i][%i] Cost: %f\n", it + 1, i, epoch_cost);
			back_prop(i);
		}
		float val_cost = 0;
		for (int i = 0; i < static_cast<int>(dataset.valY.size()); i++) {
			feed_input(i);
			forward_prop();
			val_cost += compute_cost(i);
		}
		printf("Epoch [%i] Train Cost: %f\n", it, epoch_cost / static_cast<int>(dataset.trainX.size()));
		printf("Epoch [%i] Val Cost: %f\n", it, val_cost / static_cast<int>(dataset.valY.size()));
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
			net.layers[i].neurons[j].z = 0;
			for (int k = 0; k < net.num_neurons[i - 1]; k++)
			{
				// z(current_layer) = w(current_layer-1) * a(current_layer-1)
				//printf("forward(%i) w[%i][%i]: %f\n", i , j + 1, k + 1, net.layers[i - 1].neurons[k].w[j]);
				net.layers[i].neurons[j].z += (net.layers[i - 1].neurons[k].w[j]) * (net.layers[i - 1].neurons[k].actv);
			}
			net.layers[i].neurons[j].z += net.layers[i].neurons[j].bias;

			// Sigmoid Activation function
			if (i > net.num_layers - 1) {
				net.layers[i].neurons[j].actv = sigmoid(net.layers[i].neurons[j].z);
			}
			else {
				net.layers[i].neurons[j].actv = net.layers[i].neurons[j].z;
			}
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
		// delta z(last_layer) = y'-y
		net.layers[num_layers - 1].neurons[j].dz = net.layers[num_layers - 1].neurons[j].actv - dataset.trainY[p][j];

		for (int k = 0; k < num_neurons[num_layers - 2]; k++)
		{
			// dw(last_layer-1) = delta z(last_layer) * a(last_layer - 1)
			net.layers[num_layers - 2].neurons[k].dw[j] = net.layers[num_layers - 1].neurons[j].dz * net.layers[num_layers - 2].neurons[k].actv;
		}

		// dbias(last_layer) = delta z(last_layer)
		net.layers[num_layers - 1].neurons[j].dbias = net.layers[num_layers - 1].neurons[j].dz;
	}

	// Hidden Layers
	for (int i = num_layers - 2; i > 0; i--)
	{
		for (int j = 0; j < num_neurons[i]; j++)
		{
			net.layers[i].neurons[j].dz = 0;
			for (int k = 0; k < num_neurons[i + 1]; k++) { 
				// delta z(current_layer) = w(current_layer) * delta z(current_layer + 1) * z(current_layer)*(1-z(current_layer))
				net.layers[i].neurons[j].dz += net.layers[i].neurons[j].w[k] * net.layers[i + 1].neurons[k].dz *
					net.layers[i].neurons[j].z * (1 - net.layers[i].neurons[j].z);
			}

			for (int k = 0; k < num_neurons[i - 1]; k++)
			{
				// dw(current_layer - 1) = delta z(current_layer) * a(current_layer - 1)
				net.layers[i - 1].neurons[k].dw[j] = net.layers[i].neurons[j].dz * net.layers[i - 1].neurons[k].actv;
			}

			// dbias(current_layer) = delta z(current_layer)
			net.layers[i].neurons[j].dbias = net.layers[i].neurons[j].dz;
		}
	}

	// update weights and biases
	for (int i = num_layers - 1; i > 0; i--) {
		for (int j = 0; j < num_neurons[i]; j++) {
			for (int k = 0; k < num_neurons[i - 1]; k++) {
				// w(current_layer) = w(current_layer) - lr * dw(current_layer)
				net.layers[i - 1].neurons[k].w[j] -= alpha * net.layers[i - 1].neurons[k].dw[j];
				//printf("back(%i) w[%i][%i]: %f\n", i, j+1, k+1, net.layers[i - 1].neurons[k].w[j]);
			}
			// w(current_layer) = w(current_layer) - lr * dw(current_layer)
			net.layers[i].neurons[j].bias -= alpha * net.layers[i].neurons[j].dbias;
			//printf("back(%i) bias[%i]: %f\n", i + 1, j + 1, net.layers[i].neurons[j].dbias);
		}
	}
}


float Trainer::pred(vector<float> data)
{
	for (int j = 0; j < net.num_neurons[0]; j++)
	{
		net.layers[0].neurons[j].actv = data[j];
	}
	forward_prop();
	return net.layers[net.num_layers - 1].neurons[0].actv;
}

