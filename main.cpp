// CIFAR-10 CNN demo for a vertical terminal / YouTube Short.
// ENTER: stop training and start predictions.
// ENTER again: stop the predictions and exit.
// Linux terminal (poll + termios); C++26.

#include "convolutional_neural_network.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <poll.h>
#include <termios.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {
constexpr std::size_t SIDE = 32;
constexpr std::size_t PLANE = SIDE * SIDE;
constexpr std::size_t INPUT_SIZE = 3 * PLANE;
constexpr std::size_t CLASSES = 10;
constexpr std::size_t IMAGE_WIDTH = SIDE * 2 + 2; // padding on each side

constexpr std::array<std::string_view, CLASSES> LABELS = {
    "AIRPLANE", "AUTOMOBILE", "BIRD", "CAT", "DEER",
    "DOG", "FROG", "HORSE", "SHIP", "TRUCK"
};

constexpr std::string_view RESET = "\x1b[0m";
constexpr std::string_view BOLD  = "\x1b[1m";
constexpr std::string_view PINK  = "\x1b[38;2;255;151;200m";
constexpr std::string_view LILAC = "\x1b[38;2;199;177;255m";
constexpr std::string_view MINT  = "\x1b[38;2;145;255;210m";
constexpr std::string_view WHITE = "\x1b[38;2;247;242;255m";
constexpr std::string_view MUTED = "\x1b[38;2;161;154;183m";

volatile std::sig_atomic_t interrupted = 0;
void interruptHandler(int) { interrupted = 1; }

struct TerminalGuard {
    termios saved{};
    bool changed = false;

    TerminalGuard() {
        if (::isatty(STDIN_FILENO) && ::tcgetattr(STDIN_FILENO, &saved) == 0) {
            termios settings = saved;
            settings.c_lflag &= ~ECHO; // Don't scatter echoed Enter presses over the display.
            changed = (::tcsetattr(STDIN_FILENO, TCSANOW, &settings) == 0);
        }
        std::cout << "\x1b[?25l" << std::flush;
    }

    ~TerminalGuard() {
        if (changed) ::tcsetattr(STDIN_FILENO, TCSANOW, &saved);
        std::cout << RESET << "\x1b[?25h\n" << std::flush;
    }
};

// Uses canonical terminal input: poll() becomes ready only once ENTER is pressed.
// A nonzero timeout lets ENTER interrupt the prediction slideshow immediately.
bool enterPressed(int timeoutMs = 0) {
    pollfd fd{STDIN_FILENO, POLLIN, 0};
    for (;;) {
        if (interrupted) return true;
        const int ready = ::poll(&fd, 1, timeoutMs);
        if (ready == 0) return false;
        if (ready < 0) {
            if (errno == EINTR) return interrupted != 0;
            throw std::runtime_error("poll(stdin) failed");
        }
        if (fd.revents & POLLNVAL) throw std::runtime_error("stdin is not readable");
        if (fd.revents & (POLLIN | POLLHUP)) {
            char ch = '\0';
            const auto got = ::read(STDIN_FILENO, &ch, 1);
            if (got == 0) return true; // stdin closed
            if (got < 0) {
                if (errno == EINTR) return interrupted != 0;
                throw std::runtime_error("read(stdin) failed");
            }
            if (ch == '\n' || ch == '\r') return true;
            // Discard other characters (if someone typed before pressing ENTER).
            timeoutMs = 0;
            continue;
        }
        return false;
    }
}

std::string repeat(std::string_view s, std::size_t n) {
    std::string result;
    for (std::size_t i = 0; i < n; ++i) result += s;
    return result;
}

void clearScreen() { std::cout << "\x1b[2J\x1b[H"; }

struct Image {
    std::uint8_t label = 0;
    // CIFAR-10 stores 1024 R bytes, followed by 1024 G and 1024 B.
    std::array<std::uint8_t, INPUT_SIZE> pixels{};
};

std::vector<Image> loadBatch(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open: " + path.string());

    std::vector<Image> batch;
    batch.reserve(10'000);
    for (;;) {
        char rawLabel{};
        if (!file.read(&rawLabel, 1)) {
            if (file.eof()) break;
            throw std::runtime_error("Failed reading label in " + path.string());
        }
        Image image;
        image.label = static_cast<std::uint8_t>(rawLabel);
        if (image.label >= CLASSES) throw std::runtime_error("Bad CIFAR label");
        if (!file.read(reinterpret_cast<char*>(image.pixels.data()), INPUT_SIZE))
            throw std::runtime_error("Truncated CIFAR image in " + path.string());
        batch.emplace_back(std::move(image));
    }
    if (batch.empty()) throw std::runtime_error("Empty CIFAR batch: " + path.string());
    return batch;
}

std::vector<float> normalize(const Image& img) {
    std::vector<float> result(INPUT_SIZE);
    for (std::size_t i = 0; i < INPUT_SIZE; ++i)
        result[i] = static_cast<float>(img.pixels[i]) / 255.0f;
    return result;
}

struct Metrics {
    double accuracy = 0.0;
    double loss = 0.0;
};

// Evaluate on a fixed subset never supplied to CNN::fit().
Metrics evaluate(const convolutional_neural_network& cnn,
                 const std::vector<Image>& images,
                 const std::vector<std::size_t>& indices) {
    std::size_t correct = 0;
    double crossEntropy = 0.0;
    for (auto index : indices) {
        const auto predictions = cnn.predict(normalize(images[index]), SIDE, SIDE);
        if (predictions.size() != CLASSES)
            throw std::runtime_error("CNN must output 10 probabilities");
        const auto predictedClass = static_cast<std::size_t>(
            std::max_element(predictions.begin(), predictions.end()) - predictions.begin());
        correct += (predictedClass == images[index].label);
        crossEntropy -= std::log(std::clamp(
            static_cast<double>(predictions[images[index].label]), 1e-10, 1.0));
    }
    const double count = static_cast<double>(indices.size());
    return {100.0 * correct / count, crossEntropy / count};
}

void progressBar(double fraction, std::size_t width, std::string_view fillColor) {
    const auto filled = static_cast<std::size_t>(
        std::lround(std::clamp(fraction, 0.0, 1.0) * width));
    std::cout << fillColor << repeat("━", filled)
              << MUTED << repeat("─", width - filled) << RESET;
}

void trainingScreen(std::size_t seen, std::size_t epoch,
                    Metrics metrics, std::size_t validationCount) {
    clearScreen();
    std::cout << "\n   " << PINK << BOLD << "✧  tiny vision lab  ✧" << RESET << "\n";
    std::cout << "   " << MUTED << "CIFAR-10  /  learning to see" << RESET << "\n\n\n";
    std::cout << "   " << LILAC << BOLD << "✦ TRAINING" << RESET
              << "   " << MINT << "● LIVE" << RESET << "\n\n\n";

    std::cout << "   " << MUTED << "IMAGES LEARNED" << RESET << "\n";
    std::cout << "   " << WHITE << BOLD << seen << RESET
              << "   " << MUTED << "epoch " << epoch << RESET << "\n\n\n";

    std::cout << "   " << MUTED << "VALIDATION ACCURACY" << RESET << "\n";
    std::cout << "   " << MINT << BOLD << std::fixed << std::setprecision(1)
              << metrics.accuracy << "%" << RESET << "\n";
    std::cout << "   ";
    progressBar(metrics.accuracy / 100.0, 40, MINT);
    std::cout << "\n\n\n";

    std::cout << "   " << MUTED << "CROSS-ENTROPY LOSS" << RESET << "\n";
    std::cout << "   " << WHITE << BOLD << std::fixed << std::setprecision(3)
              << metrics.loss << RESET << "\n\n";
    std::cout << "   " << MUTED << "evaluated on " << validationCount
              << " held-out images" << RESET << "\n\n\n\n";
    std::cout << "   " << PINK << BOLD << "♡ PRESS ENTER TO REVEAL PREDICTIONS" << RESET << "\n";
    std::cout << "   " << MUTED << "training continues until you do" << RESET << std::flush;
}

// A compact 5-row pixel font: still punchy, but makes the longest CIFAR class
// name fit above the confidence display without scrolling off the screen.
using Glyph = std::array<std::string_view, 5>;
Glyph glyph(char c) {
    switch (c) {
        case 'A': return {" ### ","#   #","#####","#   #","#   #"};
        case 'B': return {"#### ","#   #","#### ","#   #","#### "};
        case 'C': return {" ####","#    ","#    ","#    "," ####"};
        case 'D': return {"#### ","#   #","#   #","#   #","#### "};
        case 'E': return {"#####","#    ","#### ","#    ","#####"};
        case 'F': return {"#####","#    ","#### ","#    ","#    "};
        case 'G': return {" ####","#    ","# ###","#   #"," ####"};
        case 'H': return {"#   #","#   #","#####","#   #","#   #"};
        case 'I': return {"#####","  #  ","  #  ","  #  ","#####"};
        case 'J': return {"#####","   # ","   # ","#  # "," ##  "};
        case 'K': return {"#   #","#  # ","###  ","#  # ","#   #"};
        case 'L': return {"#    ","#    ","#    ","#    ","#####"};
        case 'M': return {"#   #","## ##","# # #","#   #","#   #"};
        case 'N': return {"#   #","##  #","# # #","#  ##","#   #"};
        case 'O': return {" ### ","#   #","#   #","#   #"," ### "};
        case 'P': return {"#### ","#   #","#### ","#    ","#    "};
        case 'R': return {"#### ","#   #","#### ","#  # ","#   #"};
        case 'S': return {" ####","#    "," ### ","    #","#### "};
        case 'T': return {"#####","  #  ","  #  ","  #  ","  #  "};
        case 'U': return {"#   #","#   #","#   #","#   #"," ### "};
        case 'V': return {"#   #","#   #","#   #"," # # ","  #  "};
        default: return {"     ","     ","     ","     ","     "};
    }
}

void bigLabel(std::string_view label) {
    const std::size_t total = label.size() * 6 - 1;
    const std::size_t left = total < IMAGE_WIDTH ? (IMAGE_WIDTH - total) / 2 : 0;
    for (std::size_t y = 0; y < 5; ++y) {
        std::cout << "   " << repeat(" ", left) << WHITE << BOLD;
        for (std::size_t i = 0; i < label.size(); ++i) {
            if (i) std::cout << ' ';
            const Glyph pattern = glyph(label[i]);
            for (const char p : pattern[y])
                std::cout << (p == '#' ? "█" : " ");
        }
        std::cout << RESET << "\n";
    }
}

// CIFAR is *natively* only 32x32. Light unsharp masking boosts existing
// edges; it does not invent detail. Rendering remains nearest-neighbor RGB.
using RGB = std::array<int, 3>;
std::array<RGB, PLANE> sharpen(const Image& image) {
    std::array<RGB, PLANE> output{};
    auto value = [&](int channel, int y, int x) -> double {
        y = std::clamp(y, 0, static_cast<int>(SIDE) - 1);
        x = std::clamp(x, 0, static_cast<int>(SIDE) - 1);
        return image.pixels[channel * PLANE + static_cast<std::size_t>(y) * SIDE + x];
    };
    constexpr double sharpenAmount = 0.85;
    for (int y = 0; y < static_cast<int>(SIDE); ++y) {
        for (int x = 0; x < static_cast<int>(SIDE); ++x) {
            for (int c = 0; c < 3; ++c) {
                double weighted = 0.0;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                        weighted += ((dy == 0 ? 2 : 1) * (dx == 0 ? 2 : 1))
                                    * value(c, y + dy, x + dx);
                const double original = value(c, y, x);
                const double blurred = weighted / 16.0;
                output[static_cast<std::size_t>(y) * SIDE + x][c] =
                    std::clamp(static_cast<int>(std::lround(
                        original + sharpenAmount * (original - blurred))), 0, 255);
            }
        }
    }
    return output;
}

void drawImage(const Image& image) {
    const auto rgb = sharpen(image);
    std::cout << "   " << LILAC << "╭" << repeat("─", IMAGE_WIDTH) << "╮" << RESET << "\n";
    for (std::size_t y = 0; y < SIDE; ++y) {
        std::cout << "   " << LILAC << "│" << RESET << " ";
        for (std::size_t x = 0; x < SIDE; ++x) {
            const auto& p = rgb[y * SIDE + x];
            std::cout << "\x1b[48;2;" << p[0] << ';' << p[1] << ';' << p[2] << "m  ";
        }
        std::cout << RESET << " " << LILAC << "│" << RESET << "\n";
    }
    std::cout << "   " << LILAC << "╰" << repeat("─", IMAGE_WIDTH) << "╯" << RESET << "\n";
}

void predictionScreen(const Image& image,
                      const std::vector<float>& probabilities,
                      std::size_t frame) {
    if (probabilities.size() != CLASSES)
        throw std::runtime_error("CNN must output 10 probabilities");
    const auto guess = static_cast<std::size_t>(
        std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin());
    const double confidence = std::clamp(static_cast<double>(probabilities[guess]), 0.0, 1.0);

    clearScreen();
    std::cout << "   " << PINK << BOLD << "✧  tiny vision lab  ✧" << RESET
              << "     " << MUTED << "picture #" << frame << RESET << "\n";
    std::cout << "   " << MUTED << "WHAT DOES MY CNN SEE?" << RESET << "\n";
    drawImage(image);
    std::cout << "   " << PINK << BOLD << "✦  MODEL PREDICTION" << RESET << "\n";
    bigLabel(LABELS[guess]);
    std::cout << "\n   " << LILAC << BOLD << "✦ CONFIDENCE  " << RESET
              << MINT << BOLD << std::fixed << std::setprecision(0)
              << confidence * 100.0 << "%" << RESET << "\n";
    std::cout << "   ";
    progressBar(confidence, 40, MINT);
    std::cout << "\n   " << MUTED << "press ENTER to finish" << RESET << std::flush;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const fs::path datasetDir = argc > 1 ? argv[1] : "cifar-10-batches-bin";
        const int frameMs = argc > 2 ? std::stoi(argv[2]) : 2300;
        if (frameMs < 0) throw std::invalid_argument("Frame delay must be nonnegative");

        std::cout << "Loading CIFAR-10...\n";
        std::vector<Image> train;
        train.reserve(50'000);
        for (int i = 1; i <= 5; ++i) {
            auto batch = loadBatch(datasetDir / ("data_batch_" + std::to_string(i) + ".bin"));
            train.insert(train.end(), std::make_move_iterator(batch.begin()),
                         std::make_move_iterator(batch.end()));
        }
        const auto test = loadBatch(datasetDir / "test_batch.bin");
        if (train.size() < 200) throw std::runtime_error("Need at least 200 training images");

        // 32x32 RGB -> 8 filters (3x3, stride 1) -> 8 feature maps 30x30
        // -> 2x2 max pooling -> 8 * 15 * 15 = 1800 MLP inputs.
        constexpr std::size_t FILTERS = 8, FILTER_SIZE = 3, FILTER_STRIDE = 1;
        constexpr std::size_t POOL_SIZE = 2, POOL_STRIDE = 2;
        constexpr std::size_t CONV_SIDE = (SIDE - FILTER_SIZE) / FILTER_STRIDE + 1;
        constexpr std::size_t POOL_SIDE = (CONV_SIDE - POOL_SIZE) / POOL_STRIDE + 1;
        constexpr std::size_t MLP_INPUT = FILTERS * POOL_SIDE * POOL_SIDE;

        convolutional_neural_network cnn(
            3, FILTERS, FILTER_SIZE, FILTER_STRIDE, POOL_SIZE, POOL_STRIDE,
            {MLP_INPUT, 80, CLASSES}, 0.0005f
        );

        std::mt19937 rng(std::random_device{}());
        std::vector<std::size_t> order(train.size());
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);

        // Fixed hold-out: these images NEVER participate in training.
        constexpr std::size_t VALIDATION_SIZE = 128;
        const std::vector<std::size_t> validation(
            order.end() - VALIDATION_SIZE, order.end());
        order.resize(order.size() - VALIDATION_SIZE);

        std::signal(SIGINT, interruptHandler);
        TerminalGuard terminal;
        std::size_t seen = 0;
        Metrics metrics = evaluate(cnn, train, validation);
        trainingScreen(seen, 1, metrics, validation.size());

        // Train without a predetermined endpoint. ENTER alone changes phases.
        while (!interrupted) {
            if (enterPressed()) break;

            const Image& image = train[order[seen % order.size()]];
            std::vector<float> target(CLASSES, 0.0f);
            target[image.label] = 1.0f;
            cnn.fit(normalize(image), SIDE, SIDE, target);
            ++seen;

            if (seen % order.size() == 0) std::shuffle(order.begin(), order.end(), rng);
            if (seen % 750 == 0) metrics = evaluate(cnn, train, validation);
            if (seen % 150 == 0) {
                trainingScreen(seen, seen / order.size() + 1, metrics, validation.size());
            }
        }
        if (interrupted) return 0;

        // Keep showing previously unseen test images until ENTER is pressed.
        std::vector<std::size_t> testOrder(test.size());
        std::iota(testOrder.begin(), testOrder.end(), 0);
        std::shuffle(testOrder.begin(), testOrder.end(), rng);
        std::size_t frame = 0;
        while (!interrupted) {
            if (enterPressed()) break;
            const Image& image = test[testOrder[frame % testOrder.size()]];
            predictionScreen(image, cnn.predict(normalize(image), SIDE, SIDE), frame + 1);
            ++frame;
            if (frame % testOrder.size() == 0) std::shuffle(testOrder.begin(), testOrder.end(), rng);
            if (enterPressed(frameMs)) break;
        }
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << "\n"
                  << "Usage: ./CNN [path-to-cifar-10-batches-bin] [frame_delay_ms=2300]\n";
        return 1;
    }
}