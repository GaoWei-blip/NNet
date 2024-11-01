#include "Dataset.h"
#include <fstream>
#include <sstream>
#include <iostream>

Dataset::Dataset(const string& filename, float trainRatio) {
	ifstream file(filename);

	if (!file.is_open()) {
		cerr << "Failed to open file: " << filename << endl;
	}

	string line;
	// Skip title line
	getline(file, line);

	while (getline(file, line)) {
		istringstream ss(line);
		StockData data;

		string temp;
		getline(ss, temp, ',');  // Skip the date field
		getline(ss, temp, ','); // Skip the timestamp field

		//ss >> data.timestamp;
		ss >> data.open;
		ss.ignore(1, ',');  
		ss >> data.high;
		ss.ignore(1, ',');  
		ss >> data.low;
		ss.ignore(1, ',');  
		ss >> data.close;
		ss.ignore(1, ',');  
		ss >> data.volume;

		dataList.push_back(data);
	}

	file.close();

	splitTrainVal(trainRatio);
}

void Dataset::splitTrainVal(float trainRatio) {

	size_t trainSize = static_cast<size_t>(dataList.size() * trainRatio);

	for (size_t i = 0; i < dataList.size(); ++i) {
		vector<float> x;
		vector<float> y;
		//x.push_back(dataList[i].timestamp);
		x.push_back(dataList[i].open);
		x.push_back(dataList[i].high);
		x.push_back(dataList[i].low);
		x.push_back(dataList[i].volume); 
		y.push_back(dataList[i].close);

		if (i < trainSize) {
			trainX.push_back(x); 
			trainY.push_back(y);
		}
		else {
			valX.push_back(x);
			valY.push_back(y);
		}
	}

}