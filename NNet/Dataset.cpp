#include "Dataset.h"
#include <fstream>
#include <sstream>
#include <iostream>

Dataset::Dataset(const string& filename) {
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
}

void Dataset::splitTrainVal(float trainRatio, bool isNorm) {

	size_t trainSize = static_cast<size_t>(dataList.size() * trainRatio);

	vector<float> maxValues(5, numeric_limits<float>::min());
	vector<float> minValues(5, numeric_limits<float>::max());

	if (isNorm) {
		for (const auto& data : dataList) {
			maxValues[0] = max(maxValues[0], data.open);
			maxValues[1] = max(maxValues[1], data.high);
			maxValues[2] = max(maxValues[2], data.low);
			maxValues[3] = max(maxValues[3], data.volume);
			maxValues[4] = max(maxValues[4], data.close);

			minValues[0] = min(minValues[0], data.open);
			minValues[1] = min(minValues[1], data.high);
			minValues[2] = min(minValues[2], data.low);
			minValues[3] = min(minValues[3], data.volume);
			minValues[4] = min(minValues[4], data.close);
		}
	}

	for (size_t i = 0; i < dataList.size(); ++i) {
		vector<float> x;
		vector<float> y;
		
		if (isNorm) {
			x.push_back((dataList[i].open - minValues[0]) / (maxValues[0] - minValues[0]));
			x.push_back((dataList[i].high - minValues[1]) / (maxValues[1] - minValues[1]));
			x.push_back((dataList[i].low - minValues[2]) / (maxValues[2] - minValues[2]));
			x.push_back((dataList[i].volume - minValues[3]) / (maxValues[3] - minValues[3]));
			y.push_back((dataList[i].close - minValues[4]) / (maxValues[4] - minValues[4]));
		}
		else
		{
			//x.push_back(dataList[i].timestamp);
			x.push_back(dataList[i].open);
			x.push_back(dataList[i].high);
			x.push_back(dataList[i].low);
			x.push_back(dataList[i].volume);
			y.push_back(dataList[i].close);
		}

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