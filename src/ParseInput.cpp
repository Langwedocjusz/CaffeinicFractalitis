#include "ParseInput.h"

#include <expected>
#include <vector>
#include <map>
#include <cctype>

#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static bool IsUint(std::string_view token);
static bool IsZero(std::string_view token);

static auto GetJobs(std::vector<std::string_view>& args) 
    -> std::expected<std::optional<uint32_t>, std::string>
{
    std::optional<uint32_t> ret = std::nullopt;

    bool already_set = false;

    for (auto it = args.begin(); it != args.end();)
    {
        bool erase = false;

        if (*it == "-j")
        {
            if (already_set)
                return std::unexpected("Cannot set -j option more than once.");

            already_set = true;

            if (it + 1 != args.end())
            {
                auto value = *(it + 1);

                if (!IsUint(value) || IsZero(value))
                {
                    return std::unexpected("Option -j can only be set to positive integer value");
                }

                ret = std::stoi(std::string(value));

                erase = true;
            }
        }

        if (erase)
            args.erase(it, it+2);
        else
            ++it;
    }

    return ret;
}

static auto GetSimd(std::vector<std::string_view>& args)
    -> std::expected<std::optional<SimdType>, std::string>
{
    std::optional<SimdType> ret = std::nullopt;

    bool already_set = false;

    const std::map<std::string, SimdType> simd_options{
        {"-Scalar", SimdType::Scalar},
        {"-SSE",    SimdType::SSE},
        {"-AVX",    SimdType::AVX},
    };

    for (auto it = args.begin(); it != args.end();)
    {
        std::string opt(*it);

        bool erase = false;

        if (simd_options.count(opt))
        {
            if (already_set)
            {
                return std::unexpected("Simd option flag can only be set once");
            }

            already_set = true;

            ret = simd_options.at(opt);

            erase = true;
        }

        if(erase) 
            args.erase(it);
        else 
            ++it;
    }

    return ret;
}

ProgramArgs ParseInput(int argc, char* argv[])
{
    ProgramArgs res;

    constexpr uint32_t max_supported_args = 4;

    if (argc > max_supported_args + 1)
    {
        res.ExitMessage = "Too many arguments provided.";
        return res;
    }
        
    std::vector<std::string_view> args(argv+1, argv+argc);

    const auto jobs = GetJobs(args);

    if (jobs.has_value())
        res.NumJobs = jobs.value();
    else
    {
        res.ExitMessage = jobs.error();
        return res;
    }
        
    const auto simd = GetSimd(args);

    if (simd.has_value())
        res.Simd = simd.value();
    else
    {
        res.ExitMessage = simd.error();
        return res;
    }

    if (args.size() == 0)
    {
        res.ExitMessage = "Missing parameter: path to json file";
        return res;
    }

    else if (args.size() > 1)
    {
        res.ExitMessage = "Only accepted arguments are a path to json file and execution flags";
        return res;
    }

    std::string filename(args[0]);

    std::ifstream file(filename);

    if (file)
    {
        try
        {
            json data = json::parse(file);

            auto RetrieveGenerator = [](const std::string& token)
            {
                const std::map<std::string, FractalGenerator> map{
                    {"SmoothIter", FractalGenerator::SmoothIter},
                    {"Gradient",   FractalGenerator::Gradient}
                };

                return map.at(token);
            };

            auto RetrieveColoring = [](const std::string& token)
            {
                using namespace Image;

                const std::map<std::string, ImageColoring> map{
                    {"IterToColorIQ",   ImageColoring::IterToColorIQ},
                    {"NormedGrayscale", ImageColoring::NormedGrayscale},
                    {"ColorHSV",        ImageColoring::ColorHSV}
                };

                return map.at(token);
            };

            res.Width = data["Image Width"];
            res.Height = data["Image Height"];
            res.NumFrames = data["Num Frames"];

            res.CenterX = data["Image Center"][0];
            res.CenterY = data["Image Center"][1];
            res.InitialWidth = data["Initial Width"];
            res.ZoomSpeed = data["Zoom Speed"];

            res.Generator = RetrieveGenerator(data["Generator"]);
            res.Coloring = RetrieveColoring(data["Coloring"]);
        }

        catch(const json::exception& e)
        {
            res.ExitMessage = "Unable to parse json file:\n" + std::string(e.what());
            return res;
        }
    }

    else
    {
        res.ExitMessage = "Failed to open file " + filename;
        return res;
    }

    return res;
}

static bool IsDigit(char ch)
{
    return std::isdigit(static_cast<unsigned char>(ch));
}

static bool IsUint(std::string_view token)
{        
    bool res = true;

    for (const char ch : token)
        res &= IsDigit(ch);

    return res;
}

static bool IsZero(std::string_view token)
{
    bool res = true;

    for (const char ch : token)
        res &= (ch == '0');

    return res;
}

