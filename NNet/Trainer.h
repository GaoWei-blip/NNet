#pragma once
#ifndef TRAINER_H
#define TRAINER_H

#include "NNet.h"
#include "Dataset.h"

class Trainer{
public:
	NNet net;
	Dataset dataset;
	int count;
	float full_cost;
	vector<float> cost;
	
	Trainer(NNet net, Dataset dataset);
	~Trainer();

	void feed_input(int i);
	void forward_prop();
	void compute_cost(int i);
	void back_prop(int p);
	//void update_weights();

	float sigmoid(int x);

	void train_neural_net();
	
};

#endif