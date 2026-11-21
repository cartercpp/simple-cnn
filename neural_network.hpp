//
// Created by cartercpp on 11/21/26.
//

#pragma once

#include <initializer_list>
#include <vector>
#include <mdspan>
#include <cstddef>

class neural_network
{
public:

    // CONSTRUCTORS

    explicit neural_network(
        std::initializer_list<std::size_t> layerSizes,
        float learningRate
    );
        //pre(layerSizes.size() >= 3)
        //pre((learningRate > 0.0f) && (learningRate < 1.0f));

    // METHODS

    [[nodiscard]] std::vector<std::vector<float>> forward(const std::vector<float>& input) const;
        //pre(input.size() == m_layerSizes[0]);

    std::vector<float> backward(
        const std::vector<std::vector<float>>& activations,
        std::vector<float> gradient
    );

    void update_weights_and_biases();
    void zero_gradients();

private:

    using Matrix = std::mdspan<const float, std::dextents<std::size_t, 2>>;

    [[nodiscard]] static float Relu(float x) noexcept;
    [[nodiscard]] static float ReluDerivative(float x) noexcept;
    [[nodiscard]] static std::vector<float> OuterProduct(
        const std::vector<float>& lArg,
        const std::vector<float>& rArg
    );

    [[nodiscard]] static std::vector<float> MatrixMatrix(Matrix lArg, Matrix rArg);
        //pre(lArg.extent(1) == rArg.extent(0));

    [[nodiscard]] static std::vector<float> MatrixVector(Matrix mat, const std::vector<float>& vec);
        //pre(mat.extent(1) == vec.size());

    [[nodiscard]] static std::vector<float> VectorMatrix(const std::vector<float>& vec, Matrix mat);
        //pre(vec.size() == mat.extent(0));

    std::vector<std::vector<float>> m_weights,
                                    m_weightGradients,
                                    m_biases,
                                    m_biasGradients;
    std::vector<std::size_t> m_layerSizes;
    float m_learningRate;
};