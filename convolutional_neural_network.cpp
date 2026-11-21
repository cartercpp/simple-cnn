//
// Created by cartercpp on 11/21/26.
//

#include "convolutional_neural_network.hpp"
#include <vector>
#include <mdspan>
#include <algorithm>
#include <ranges>
#include <random>
#include <limits>
#include <cstddef>
#include <cmath>

std::vector<std::vector<float>> convolutional_neural_network::FeatureMaps(
    const std::vector<float>& image,
    std::size_t imageRows,
    std::size_t imageColumns
) const
{
    std::vector<std::vector<float>> featureMaps(m_numFilters);

    for (std::size_t filter = 0; filter < m_numFilters; ++filter)
    {
        featureMaps[filter].reserve(
            ((imageRows - m_filterSize) / m_filterStride + 1) * ((imageColumns - m_filterSize) / m_filterStride + 1)
        );

        for (std::size_t imageRow = 0; imageRow + m_filterSize <= imageRows; imageRow += m_filterStride)
            for (std::size_t imageColumn = 0; imageColumn + m_filterSize <= imageColumns; imageColumn += m_filterStride)
            {
                float activation = m_biases[filter];

                for (std::size_t inputChannel = 0; inputChannel < m_numInputChannels; ++inputChannel)
                    for (std::size_t iter = 0; iter < m_filterSize * m_filterSize; ++iter)
                    {
                        const std::size_t row = imageRow + iter / m_filterSize,
                                          column = imageColumn + iter % m_filterSize;

                        const float input = std::mdspan(
                            image.data(),
                            m_numInputChannels,
                            imageRows,
                            imageColumns
                        )[inputChannel, row, column];

                        const float weight = std::mdspan(
                            m_filters[filter].data(),
                            m_numInputChannels,
                            m_filterSize,
                            m_filterSize
                        )[inputChannel, iter / m_filterSize, iter % m_filterSize];

                        activation += input * weight;
                    }

                activation = (activation > 0.0f) ? activation : 0.0f;
                featureMaps[filter].push_back(activation);
            }
    }

    return featureMaps;
}

std::vector<std::vector<std::size_t>> convolutional_neural_network::MaxPoolIndices(
    const std::vector<std::vector<float>>& featureMaps,
    std::size_t featureMapRows,
    std::size_t featureMapColumns
) const
{
    std::vector<std::vector<std::size_t>> maxPoolIndices(m_numFilters);

    for (std::size_t filter = 0; filter < m_numFilters; ++filter)
    {
        maxPoolIndices[filter].reserve(
            ((featureMapRows - m_poolSize) / m_poolStride + 1) * ((featureMapColumns - m_poolSize) / m_poolStride + 1)
        );

        for (std::size_t featureMapRow = 0; featureMapRow + m_poolSize <= featureMapRows; featureMapRow += m_poolStride)
            for (std::size_t featureMapColumn = 0; featureMapColumn + m_poolSize <= featureMapColumns; featureMapColumn += m_poolStride)
            {
                float maxValue = std::numeric_limits<float>::lowest();
                std::size_t maxIndex = 0;

                for (std::size_t iter = 0; iter < m_poolSize * m_poolSize; ++iter)
                {
                    const std::size_t row = featureMapRow + iter / m_poolSize,
                                      column = featureMapColumn + iter % m_poolSize;

                    const float value = std::mdspan(
                        featureMaps[filter].data(),
                        featureMapRows,
                        featureMapColumns
                    )[row, column];

                    if (value > maxValue)
                    {
                        maxValue = value;
                        maxIndex = row * featureMapColumns + column;
                    }
                }

                maxPoolIndices[filter].push_back(maxIndex);
            }
    }

    return maxPoolIndices;
}

std::vector<float> convolutional_neural_network::predict(
    const std::vector<float>& image,
    std::size_t imageRows,
    std::size_t imageColumns
) const
{
    const std::vector<std::vector<float>> featureMaps{FeatureMaps(image, imageRows, imageColumns)};
    const std::size_t featureMapRows = (imageRows - m_filterSize) / m_filterStride + 1,
                      featureMapColumns = (imageColumns - m_filterSize) / m_filterStride + 1;
    const std::vector<std::vector<std::size_t>> maxPoolIndices{MaxPoolIndices(featureMaps, featureMapRows, featureMapColumns)};

    std::vector<float> predictorInput;
    predictorInput.reserve(m_numFilters * maxPoolIndices[0].size());

    for (std::size_t filter = 0; filter < m_numFilters; ++filter)
        for (const std::size_t maxIndex : maxPoolIndices[filter])
            predictorInput.emplace_back(featureMaps[filter][maxIndex]);

    return m_predictor.forward(predictorInput).back();
}

void convolutional_neural_network::fit(
    const std::vector<float>& image,
    std::size_t imageRows,
    std::size_t imageColumns,
    const std::vector<float>& targetProbabilities
)
{
    const std::vector<std::vector<float>> featureMaps{FeatureMaps(image, imageRows, imageColumns)};
    const std::size_t featureMapRows = (imageRows - m_filterSize) / m_filterStride + 1,
                      featureMapColumns = (imageColumns - m_filterSize) / m_filterStride + 1;
    const std::vector<std::vector<std::size_t>> maxPoolIndices{MaxPoolIndices(featureMaps, featureMapRows, featureMapColumns)};

    std::vector<float> predictorInput;
    predictorInput.reserve(m_numFilters * maxPoolIndices[0].size());
    for (std::size_t filter = 0; filter < m_numFilters; ++filter)
        for (const std::size_t maxIndex : maxPoolIndices[filter])
            predictorInput.emplace_back(featureMaps[filter][maxIndex]);

    const std::vector<std::vector<float>> predictorActivations{m_predictor.forward(predictorInput)};
    const std::vector<float>& predictedProbabilities{predictorActivations.back()};

    std::vector<float> outputGradients(m_outputSize, 0.0f);
    for (std::size_t i = 0; i < m_outputSize; ++i)
        outputGradients[i] = predictedProbabilities[i] - targetProbabilities[i];

    const std::vector<float> pooledGradients{m_predictor.backward(predictorActivations, outputGradients)};
    m_predictor.update_weights_and_biases();

    std::vector<std::vector<float>> convolutionGradients(
        m_numFilters,
        std::vector<float>(featureMapRows * featureMapColumns, 0.0f)
    );

    for (std::size_t filter = 0; filter < m_numFilters; ++filter)
    {
        const std::size_t pooledSize = maxPoolIndices[filter].size();

        for (std::size_t i = 0; i < pooledSize; ++i)
            convolutionGradients[filter][maxPoolIndices[filter][i]] += pooledGradients[filter * pooledSize + i];
    }

    for (std::size_t filter = 0; filter < m_numFilters; ++filter)
        for (std::size_t featureMapRow = 0; featureMapRow < featureMapRows; ++featureMapRow)
            for (std::size_t featureMapColumn = 0; featureMapColumn < featureMapColumns; ++featureMapColumn)
                std::mdspan(
                    convolutionGradients[filter].data(),
                    featureMapRows,
                    featureMapColumns
                )[featureMapRow, featureMapColumn]
                    *= (std::mdspan(featureMaps[filter].data(), featureMapRows, featureMapColumns)[featureMapRow, featureMapColumn] > 0.0f)
                        ? 1.0f : 0.0f;

    std::vector<std::vector<float>> filterGradients(m_numFilters,
                                        std::vector<float>(m_numInputChannels * m_filterSize * m_filterSize, 0.0f));
    std::vector<float> biasGradients(m_numFilters, 0.0f);

    for (std::size_t filter = 0; filter < m_numFilters; ++filter)
        for (std::size_t imageRow = 0; imageRow + m_filterSize <= imageRows; imageRow += m_filterStride)
            for (std::size_t imageColumn = 0; imageColumn + m_filterSize <= imageColumns; imageColumn += m_filterStride)
            {
                const float gradient = std::mdspan(
                    convolutionGradients[filter].data(),
                    featureMapRows,
                    featureMapColumns
                )[imageRow / m_filterStride, imageColumn / m_filterStride];

                biasGradients[filter] += gradient;

                for (std::size_t inputChannel = 0; inputChannel < m_numInputChannels; ++inputChannel)
                    for (std::size_t iter = 0; iter < m_filterSize * m_filterSize; ++iter)
                    {
                        const std::size_t row = imageRow + iter / m_filterSize,
                                          column = imageColumn + iter % m_filterSize;

                        const float pixel = std::mdspan(
                            image.data(),
                            m_numInputChannels,
                            imageRows,
                            imageColumns
                        )[inputChannel, row, column];

                        std::mdspan(
                            filterGradients[filter].data(),
                            m_numInputChannels,
                            m_filterSize,
                            m_filterSize
                        )[inputChannel, iter / m_filterSize, iter % m_filterSize] += pixel * gradient;
                    }
            }

    for (std::size_t filter = 0; filter < m_numFilters; ++filter)
        for (std::size_t inputChannel = 0; inputChannel < m_numInputChannels; ++inputChannel)
            for (std::size_t iter = 0; iter < m_filterSize * m_filterSize; ++iter)
                std::mdspan(
                    m_filters[filter].data(),
                    m_numInputChannels,
                    m_filterSize,
                    m_filterSize
                )[inputChannel, iter / m_filterSize, iter % m_filterSize]
                    -= m_learningRate * std::mdspan(filterGradients[filter].data(), m_numInputChannels, m_filterSize, m_filterSize)
                                            [inputChannel, iter / m_filterSize, iter % m_filterSize];

    for (std::size_t filter = 0; filter < m_numFilters; ++filter)
        m_biases[filter] -= m_learningRate * biasGradients[filter];
}

convolutional_neural_network::convolutional_neural_network(
    std::size_t numInputChannels,
    std::size_t numFilters,
    std::size_t filterSize,
    std::size_t filterStride,
    std::size_t poolSize,
    std::size_t poolStride,
    std::initializer_list<std::size_t> predictorLayerSizes,
    float learningRate
)
    : m_numInputChannels{numInputChannels},
    m_numFilters{numFilters},
    m_filterSize{filterSize},
    m_filterStride{filterStride},
    m_poolSize{poolSize},
    m_poolStride{poolStride},
    m_predictor{predictorLayerSizes, learningRate},
    m_learningRate{learningRate},
    m_outputSize{*(predictorLayerSizes.end() - 1)}
{
    std::mt19937_64 rng{std::random_device{}()};
    std::normal_distribution<float> dist(
        0.0f,
        std::sqrtf(2.0f / static_cast<float>(m_numInputChannels * m_filterSize * m_filterSize))
    );

    m_filters.resize(m_numFilters, std::vector<float>(m_numInputChannels * m_filterSize * m_filterSize, 0.0f));

    for (std::size_t filter = 0; filter < m_numFilters; ++filter)
        std::ranges::for_each(m_filters[filter], [&](float& valueRef){valueRef = dist(rng);});

    m_biases.resize(m_numFilters, 0.0f);
}
