//
// Created by cartercpp on 11/21/26.
//

#include "neural_network.hpp"
#include <initializer_list>
#include <vector>
#include <mdspan>
#include <random>
#include <ranges>
#include <algorithm>
#include <utility>
#include <cmath>
#include <cstddef>

std::vector<float> neural_network::MatrixMatrix(Matrix lArg, Matrix rArg)
{
    const std::size_t rows = lArg.extent(0),
                      columns = rArg.extent(1),
                      iterCount = lArg.extent(1);

    std::vector<float> output(rows * columns, 0.0f);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column)
            for (std::size_t iter = 0; iter < iterCount; ++iter)
                std::mdspan(output.data(), rows, columns)[row, column] += lArg[row, iter] * rArg[iter, column];

    return output;
}

std::vector<float> neural_network::MatrixVector(Matrix mat, const std::vector<float>& vec)
{
    const std::size_t rows = mat.extent(0),
                      iterCount = vec.size();

    std::vector<float> output(rows, 0.0f);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t iter = 0; iter < iterCount; ++iter)
            output[row] += mat[row, iter] * vec[iter];

    return output;
}

std::vector<float> neural_network::VectorMatrix(const std::vector<float>& vec, Matrix mat)
{
    const std::size_t columns = mat.extent(1),
                      iterCount = vec.size();

    std::vector<float> output(columns, 0.0f);
    for (std::size_t column = 0; column < columns; ++column)
        for (std::size_t iter = 0; iter < iterCount; ++iter)
            output[column] += mat[iter, column] * vec[iter];

    return output;
}

float neural_network::Relu(float x) noexcept
{
    return (x > 0.0f) ? x : 0.0f;
}

float neural_network::ReluDerivative(float x) noexcept
{
    return (x > 0.0f) ? 1.0f : 0.0f;
}

std::vector<float> neural_network::OuterProduct(
    const std::vector<float>& lArg,
    const std::vector<float>& rArg
)
{
    std::vector<float> output(lArg.size() * rArg.size(), 0.0f);

    for (std::size_t i = 0; i < lArg.size(); ++i)
        for (std::size_t i2 = 0; i2 < rArg.size(); ++i2)
            std::mdspan(output.data(), lArg.size(), rArg.size())[i, i2] = lArg[i] * rArg[i2];

    return output;
}

std::vector<std::vector<float>> neural_network::forward(const std::vector<float>& input) const
{
    std::vector<std::vector<float>> activations;
    activations.reserve(m_layerSizes.size());
    activations.push_back(input);

    for (std::size_t weightIndex = 0; weightIndex < m_weights.size(); ++weightIndex)
    {
        std::vector<float> layerActivations = MatrixVector(
            Matrix{m_weights[weightIndex].data(), m_layerSizes[weightIndex + 1], m_layerSizes[weightIndex]},
            activations[weightIndex]
        );

        for (std::size_t neuron = 0; neuron < m_layerSizes[weightIndex + 1]; ++neuron)
            layerActivations[neuron] += m_biases[weightIndex][neuron];

        if (weightIndex + 1 < m_weights.size())
        {
            std::ranges::for_each(layerActivations, [](float& activationRef){activationRef = Relu(activationRef);});
        }
        else
        {
            const float maxActivation = std::ranges::max(layerActivations);
            float softmaxSum = 0.0f;

            std::ranges::for_each(layerActivations, [&](float& activationRef) {
                activationRef = std::exp(activationRef - maxActivation);
                softmaxSum += activationRef;
            });

            std::ranges::for_each(layerActivations, [&](float& activationRef){activationRef /= softmaxSum;});
        }

        activations.emplace_back(std::move(layerActivations));
    }

    return activations;
}

std::vector<float> neural_network::backward(
    const std::vector<std::vector<float>>& activations,
    std::vector<float> gradient
)
{
    for (std::size_t weightIndex = m_weights.size(); weightIndex-- > 0;)
    {
        // make gradient be wrt preactivations:
        if (weightIndex < m_weights.size() - 1)
            for (std::size_t neuron = 0; neuron < m_layerSizes[weightIndex + 1]; ++neuron)
                gradient[neuron] *= ReluDerivative(activations[weightIndex + 1][neuron]);

        const std::vector<float> weightGradient = OuterProduct(gradient, activations[weightIndex]);
        for (std::size_t neuron = 0; neuron < m_layerSizes[weightIndex + 1] * m_layerSizes[weightIndex]; ++neuron)
            m_weightGradients[weightIndex][neuron] += weightGradient[neuron];

        for (std::size_t neuron = 0; neuron < m_layerSizes[weightIndex + 1]; ++neuron)
            m_biasGradients[weightIndex][neuron] += gradient[neuron];

        gradient = VectorMatrix(
            gradient,
            Matrix{m_weights[weightIndex].data(), m_layerSizes[weightIndex + 1], m_layerSizes[weightIndex]}
        );
    }

    return gradient;
}

void neural_network::update_weights_and_biases()
{
    for (std::size_t weightIndex = 0; weightIndex < m_weights.size(); ++weightIndex)
    {
        for (std::size_t neuron = 0; neuron < m_layerSizes[weightIndex + 1] * m_layerSizes[weightIndex]; ++neuron)
        {
            m_weights[weightIndex][neuron] -= m_learningRate * m_weightGradients[weightIndex][neuron];
            m_weightGradients[weightIndex][neuron] = 0.0f;
        }

        for (std::size_t neuron = 0; neuron < m_layerSizes[weightIndex + 1]; ++neuron)
        {
            m_biases[weightIndex][neuron] -= m_learningRate * m_biasGradients[weightIndex][neuron];
            m_biasGradients[weightIndex][neuron] = 0.0f;
        }
    }
}

void neural_network::zero_gradients()
{
    for (std::size_t weightIndex = 0; weightIndex < m_weights.size(); ++weightIndex)
    {
        std::ranges::for_each(m_weightGradients[weightIndex], [](float& gradientRef){gradientRef = 0.0f;});
        std::ranges::for_each(m_biasGradients[weightIndex], [](float& gradientRef){gradientRef = 0.0f;});
    }
}

neural_network::neural_network(std::initializer_list<std::size_t> layerSizes, float learningRate)
    : m_layerSizes{layerSizes}, m_learningRate{learningRate}
{
    m_weights.reserve(m_layerSizes.size() - 1);
    m_weightGradients.reserve(m_layerSizes.size() - 1);
    m_biases.reserve(m_layerSizes.size() - 1);
    m_biasGradients.reserve(m_layerSizes.size() - 1);

    std::mt19937_64 rng{std::random_device{}()};

    for (std::size_t layer = 1; layer < m_layerSizes.size(); ++layer)
    {
        const std::size_t layerSize = m_layerSizes[layer],
                          prevLayerSize = m_layerSizes[layer - 1];

        std::normal_distribution<float> dist(0.0f, std::sqrtf(2.0f / static_cast<float>(prevLayerSize)));

        std::vector<float> layerWeights(layerSize * prevLayerSize, 0.0f);
        std::ranges::for_each(layerWeights, [&](float& weightRef){weightRef = dist(rng);});

        m_weights.emplace_back(std::move(layerWeights));
        m_weightGradients.emplace_back(layerSize * prevLayerSize, 0.0f);
        m_biases.emplace_back(layerSize, 0.0f);
        m_biasGradients.emplace_back(layerSize, 0.0f);
    }
}