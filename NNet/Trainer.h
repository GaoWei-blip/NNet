#pragma once
#ifndef TRAINER_H
#define TRAINER_H

#include "NNet.h"
#include "Dataset.h"

class Trainer{
public:
	NNet net;
	Dataset dataset;
	vector<float> cost;
	float alpha;
	int epoch;
	
	Trainer(NNet net, Dataset dataset, int epoch=100, float alpha=0.001f);
	~Trainer();

	void feed_input(int i);
	void forward_prop();
	float compute_cost(int i);
	void back_prop(int p);

	float sigmoid(float x);

	void train_neural_net();

	float pred(vector<float> data);
	
};

#endif