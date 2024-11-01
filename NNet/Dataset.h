#pragma once
#ifndef DATASET_H
#define DATASET_H

#include "vector"
#include "string"
using namespace std;

struct StockData {
	float timestamp;
	float open;      //opening price
	float high;      //highest price
	float low;       //lowest price
	float close;     //closing price
	float volume;    //trading volume
};

class Dataset {
public:

	vector<StockData> dataList;
	vector<vector<float>> trainX;
	vector<vector<float>> trainY;
	vector<vector<float>> valX;
	vector<vector<float>> valY;

	Dataset(const string& filename, float trainRatio=0.8);
	~Dataset() {};

	void splitTrainVal(float trainRatio);

};


#endif
