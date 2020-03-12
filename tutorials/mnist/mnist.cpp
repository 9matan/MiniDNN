#define _CRT_SECURE_NO_WARNINGS

#include <fstream>
#include <algorithm>
#include "external/csv-parser/parser.hpp"
#include "MiniDNN.h"
#include "Profiler.h"

using namespace MiniDNN;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;
typedef std::pair<Matrix, Matrix> DataSet;

DataSet ReadDataSetFromCSV(std::string const& fileName, unsigned int const classesCount)
{
    PROFILE_TIME("ReadDataSetFromCSV");

    DataSet res;
    std::ifstream ifileStream(fileName);
    if (!ifileStream.good())
    {
        std::cerr << "Invalid path to a csv file!\n";
        return res;
    }

    aria::csv::CsvParser csvParser(ifileStream);

    if (csvParser.empty())
    {
        std::cerr << "Empty csv file!\n";
        return res;
    }

    auto rowIter = csvParser.begin();
    auto const header = *rowIter;
    // 1st col is a lable (output)
    size_t const featuresCount = header.size() - 1;

    std::vector<Vector> observationsInput;
    std::vector<Vector> observationsOutput;
    observationsInput.reserve(1000);
    observationsOutput.reserve(1000);
    unsigned int readRows = 0;
    for (++rowIter; rowIter != csvParser.end(); ++rowIter)
    {
        auto const& row = *rowIter;
        auto fieldIter = row.begin();
        Vector output = Vector::Zero(classesCount);
        output[std::stoi(*fieldIter)] = 1.0f;
        observationsOutput.push_back(output);

        Vector observation(featuresCount);
        for (++fieldIter; fieldIter != row.end(); ++fieldIter)
        {
            observation[fieldIter - row.begin() - 1] = std::stoi(*fieldIter);
        }
        observationsInput.push_back(observation);

        ++readRows;
        if (readRows == 1000)
        {
            readRows = 0;
            std::cerr << observationsInput.size() << "\n";
        }
    }
    assert(observationsInput.size() == observationsOutput.size());
    res.first = Matrix::Random(featuresCount, observationsInput.size());
    for (unsigned int colIndex = 0; colIndex < observationsInput.size(); ++colIndex)
    {
        res.first.col(colIndex) = observationsInput[colIndex];
    }
    res.second = Matrix::Random(classesCount, observationsOutput.size());
    for (unsigned int colIndex = 0; colIndex < observationsOutput.size(); ++colIndex)
    {
        res.second.col(colIndex) = observationsOutput[colIndex];
    }
    return res;
}

float CalcMulticlassAccuracy(Matrix const& expectedOutput, Matrix const& output)
{
    assert(expectedOutput.cols() == output.cols());
    assert(expectedOutput.rows() == output.rows());
    if (expectedOutput.rows() == 0 || expectedOutput.cols() == 0)
    {
        return 0.0f;
    }

    unsigned int truePositiveCount = 0;
    for (unsigned int observationIndex = 0; observationIndex < expectedOutput.cols(); ++observationIndex)
    {
        auto const expectedOutputArray = expectedOutput.col(observationIndex).array();
        unsigned int const expectedClass = std::max_element(expectedOutputArray.begin(), expectedOutputArray.end()) - expectedOutputArray.begin();
        auto const outputArray = output.col(observationIndex).array();
        unsigned int const predictedClass = std::max_element(outputArray.begin(), outputArray.end()) - outputArray.begin();
        if (expectedClass == predictedClass)
        {
            ++truePositiveCount;
        }
    }

    return (float)truePositiveCount / expectedOutput.cols();
}

bool IsValidData(DataSet const& data)
{
    return data.first.cols() > 0 && data.first.rows() > 0 && data.second.cols() > 0;
}

int main()
{
    unsigned int const labelsCount = 10;

    std::cout << "Reading training data ... \n";
    auto const trainingData = ReadDataSetFromCSV(std::string(MNIST_DATA_FOLDER) + "\\mnist_train.csv", labelsCount);
    if (!IsValidData(trainingData))
    {
        std::cerr << "Training data is not valid\n";
        return 0;
    }
    std::cout << "Done!\n";
    
    std::cout << "Reading test data ... \n";
    auto const testData = ReadDataSetFromCSV(std::string(MNIST_DATA_FOLDER) + "\\mnist_test.csv", labelsCount);
    if (!IsValidData(testData))
    {
        std::cerr << "Test data is not valid\n";
        return 0;
    }
    std::cout << "Done!\n";

    unsigned int const inputLayerSize = trainingData.first.rows();
    unsigned int const outputLayerSize = trainingData.second.rows();
    unsigned int const hiddenLayerSize = 30;

    Network net;
    net.add_layer(new FullyConnected<Identity>(inputLayerSize, hiddenLayerSize));
    net.add_layer(new FullyConnected<ReLU>(hiddenLayerSize, hiddenLayerSize));
    net.add_layer(new FullyConnected<Softmax>(hiddenLayerSize, outputLayerSize));
    net.set_output(new MultiClassEntropy());
    VerboseCallback callback;
    net.set_callback(callback);
    RMSProp opt;
    opt.m_lrate = 0.001; 
    const int seed = 123;
    net.init(0, 0.01, seed);
    {
        PROFILE_TIME("Fit model");
        const int batch_size = 100;
        const int epochs_count = 5;
        net.fit(opt, trainingData.first, trainingData.second, batch_size, epochs_count, seed);
    }
    {
        PROFILE_TIME("Test model");
        Matrix const testPred = net.predict(testData.first);
        float const accuracy = CalcMulticlassAccuracy(testData.second, testPred);
        std::cerr << "Accuracy: " << accuracy * 100.0f << "%\n";
    }

    return 0;
}