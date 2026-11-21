//
// Created by cartercpp on 11/21/26.
//

#pragma once

#include <initializer_list>
#include <vector>
#include <cstddef>
#include "neural_network.hpp"

class convolutional_neural_network
{
public:

    // CONSTRUCTORS

    explicit convolutional_neural_network(
        std::size_t numInputChannels,
        std::size_t numFilters,
        std::size_t filterSize,
        std::size_t filterStride,
        std::size_t poolSize,
        std::size_t poolStride,
        std::initializer_list<std::size_t> predictorLayerSizes,
        float learningRate
    );

    // METHODS

    std::vector<float> predict(
        const std::vector<float>& image,
        std::size_t imageRows,
        std::size_t imageColumns
    ) const;
        //pre(image.size() == m_numInputChannels * imageRows * imageColumns);

    void fit(
        const std::vector<float>& image,
        std::size_t imageRows,
        std::size_t imageColumns,
        const std::vector<float>& targetProbabilities
    );
        //pre(image.size() == m_numInputChannels * imageRows * imageColumns);

private:

    std::vector<std::vector<float>> FeatureMaps(
        const std::vector<float>& image,
        std::size_t imageRows,
        std::size_t imageColumns
    ) const;
        //pre(image.size() == m_numInputChannels * imageRows * imageColumns);

    std::vector<std::vector<std::size_t>> MaxPoolIndices(
        const std::vector<std::vector<float>>& featureMaps,
        std::size_t featureMapRows,
        std::size_t featureMapColumns
    ) const;
        //pre(featureMaps.size() == m_numFilters);

    neural_network m_predictor;
    std::vector<std::vector<float>> m_filters;
    std::vector<float> m_biases;
    std::size_t m_numFilters,
                m_numInputChannels,
                m_filterSize,
                m_filterStride,
                m_poolSize,
                m_poolStride,
                m_outputSize;
    float m_learningRate;
};