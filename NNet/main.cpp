#include "NNet.h"
#include "Layer.h"
#include "Neuron.h"
#include "Dataset.h"
#include "Trainer.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

using namespace std;


int main(void) {

	int num_layers = 0;           // the number of layers in the neural network
	int num_hidden_layers;    // the number of hidden layers in the neural network
	vector<int> num_neurons;  // the number of neurons in each layer

	//float alpha;                                      // learning rate
	//float* cost;                                      // the cost or loss value
	//float full_cost;                                  // the full cost value

	// ===============================================================
	
	// 1.Reading Dataset
	Dataset dataset = Dataset("../stock_price_data.csv");
	vector<vector<float>> trainX = dataset.trainX;
	vector<vector<float>> trainY = dataset.trainY;
	vector<vector<float>> valX = dataset.valX;
	vector<vector<float>> valY = dataset.valY;

	cout << "The number of dataset:" << dataset.dataList.size() << endl;
	cout << "The number of trainX:" << trainX.size() << endl;
	cout << "The number of trainY:" << trainY.size() << endl;
	cout << "The number of valX:" << valX.size() << endl;
	cout << "The number of valY:" << valY.size() << endl;

	// print a piece of data
	for (int i = 0; i < trainX[0].size(); ++i) {
		cout << "The number of trainX[0][" << i << "]:" << trainX[0][i] << endl;
	}
	cout << "The number of trainY[0]:" << trainY[0][0] << endl;

	// ===============================================================

	// 2.Input parameter and Build a network and initialize weights
	cout << "Enter the number of Hidden Layers in Neural Network:" << endl;
	cin >> num_hidden_layers;

	num_layers = num_hidden_layers + 2;
	num_neurons.resize(num_layers);
	num_neurons[0] = trainX[0].size();
	for (int i = 1; i < num_layers-1; i++) {
		cout << "Enter number of neurons in hidden layer[" << i << "]: " << endl;
		cin >> num_neurons[i];
	}
	num_neurons[num_layers - 1] = valY[0].size();
	cout << endl;

	NNet net = NNet(num_layers, num_neurons);

	// ===============================================================
	
	// 3.Train
	Trainer trainer = Trainer(net, dataset);
	printf("======================Start training...====================== \n");
	trainer.train_neural_net();



	// ===============================================================

	// 4.Test
	

}

