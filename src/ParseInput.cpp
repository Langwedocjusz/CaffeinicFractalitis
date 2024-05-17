#include "ParseInput.h"

#include <vector>
#include <cctype>

static bool IsUint(std::string_view token);
static bool IsZero(std::string_view token);

ProgramArgs ParseInput(int argc, char* argv[])
{
    ProgramArgs res;

    constexpr uint32_t supported_args = 2;

    if (argc > supported_args + 1)
    {
        res.ExitMessage = "Too many arguments provided.";
        return res;
    }
        
    const std::vector<std::string_view> args(argv+1, argv+argc);

    bool num_jobs_set = false;

    for (size_t i=0; i<args.size(); i++)
    {
        if (args[i] == "-j")
        {
            if (num_jobs_set)
            {
                res.ExitMessage = "Cannot set -j option more than once.";
                return res;
            }

            if (!IsUint(args[i+1]) || IsZero(args[i+1]))
            {
                res.ExitMessage = "Option -j can only be set to positive integer value";
                return res;
            }

            const uint32_t num_jobs = std::stoi(std::string(args[i+1]));

            res.NumJobs = num_jobs;
        }
    }

    return res;
}

static bool IsUint(std::string_view token)
{        
    bool res = true;

    for (const char ch : token)
        res &= isdigit(ch);

    return res;
}

static bool IsZero(std::string_view token)
{
    bool res = true;

    for (const char ch : token)
        res &= (ch == '0');

    return res;
}

