
/**
 * @file logger.hpp
 * @version 0.0.4
 * @brief Async Logger for C++ with timestamp, name, and configurable options.
 *
 * Environment variables:
 * - ASYNC_TRACE: Enables trace for specific logger name.
 * - ASYNC_TRACE_VERBOSE: Enables verbose trace for specific logger name.
 * - ASYNC_TRACE_MAX_BUFFER: Sets the maximum buffer size for log entries.
 * - ASYNC_TRACE_OUTPUT_DIR: Sets the output directory for log files.
 * - ASYNC_TRACE_CONSOLE: Enables logging to the console (stdout).
 *
 * Example usage:
 * @code
 * #include "logger.hpp"
 *
 * int main() {
 *     AsyncLogger logger("CUD");
 *
 *     StartTraceInfo(logger) << "This is an info message" << '\n';
 *     StartTraceVerbose(logger) << "This is a verbose message" << '\n';
 *
 *     return 0;
 * }
 * @endcode
 *
 * Async Logger for C++
 * with timestamp and name
 * configurable via environment variables.
 *
 *
 * @author Wassim KABALAN
 */

#ifndef ASYNC_LOGGER_HPP
#define ASYNC_LOGGER_HPP

#include <chrono>
#include <cstdlib>
#include <cxxabi.h>
#include <execinfo.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>

#ifdef MPI_VERSION
#include <mpi.h>
#endif

class AsyncLogger {
public:
    AsyncLogger(const std::string &name)
            : name(name),
              bufferSize(10 * 1024 * 1024),
              buffer(""),
              traceInfo(false),
              traceVerbose(false),
              traceToConsole(true) {
        static const char *traceEnv = std::getenv("ASYNC_TRACE");
        if (traceEnv != nullptr) {
            std::string traceString = traceEnv;
            size_t pos = traceString.find(name);
            if (pos != std::string::npos) {
                traceInfo = true;
            }
        }

        static const char *traceEnvVerb = std::getenv("ASYNC_TRACE_VERBOSE");
        if (traceEnvVerb != nullptr) {
            std::string traceString = traceEnvVerb;
            size_t pos = traceString.find(name);
            if (pos != std::string::npos) {
                traceInfo = true;
                traceVerbose = true;
            }
        }

        static const char *bufferSizeEnv = std::getenv("ASYNC_TRACE_MAX_BUFFER");
        if (bufferSizeEnv != nullptr) {
            bufferSize = std::atoi(bufferSizeEnv);
        }

        static const char *outputDirEnv = std::getenv("ASYNC_TRACE_OUTPUT_DIR");
        if (outputDirEnv != nullptr) {
            outputDir = outputDirEnv;
            // Ensure the output directory exists
            // std::filesystem::create_directories(outputDir);
            traceToConsole = false;
        }

        static const char *traceToConsoleEnv = std::getenv("ASYNC_TRACE_CONSOLE");
        if (traceToConsoleEnv != nullptr) {
            traceToConsole = std::atoi(traceToConsoleEnv) != 0;
            traceToConsole = true;
        }
        static const char *nobufferEnv = std::getenv("ASYNC_TRACE_NOBUFFER");
        if (nobufferEnv != nullptr) {
            nobuffer = std::atoi(nobufferEnv) != 0;
            nobuffer = true;
        }

#ifdef MPI_VERSION
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    }

    AsyncLogger &startTraceInfo() {
        if (traceInfo || traceVerbose) {
            std::ostringstream ss;
            addTimestamp(ss);
            ss << "[INFO] ";
            ss << "[" << name << "] ";
            buffer += ss.str();
        }
        return *this;
    }

    AsyncLogger &startTraceVerbose() {
        if (traceInfo || traceVerbose) {
            std::ostringstream ss;
            addTimestamp(ss);
            ss << "[VERB] ";
            ss << "[" << name << "] ";
            buffer += ss.str();
        }
        return *this;
    }

    template <typename T>
    AsyncLogger &operator<<(const T &value) {
        if (traceInfo || traceVerbose) {
            std::ostringstream ss;
            ss << value;
            buffer += ss.str();
            if (buffer.size() >= bufferSize || nobuffer) {
                flush();
            }
        }
        return *this;
    }

    // Specialization for bool
    AsyncLogger &operator<<(bool value) {
        if (traceInfo || traceVerbose) {
            std::ostringstream ss;
            ss << std::boolalpha << value;
            buffer += ss.str();
            if (buffer.size() >= bufferSize || nobuffer) {
                flush();
            }
        }
        return *this;
    }

    // Specialization for std::endl
    AsyncLogger &operator<<(std::ostream &(*manipulator)(std::ostream &)) {
        if (traceInfo || traceVerbose) {
            std::ostringstream ss;
            ss << manipulator;
            buffer += ss.str();
            if (buffer.size() >= bufferSize || nobuffer) {
                flush();
            }
        }
        return *this;
    }

    ~AsyncLogger() {
        if (traceInfo || traceVerbose) {
            flush();
        }
    }

    bool getTraceInfo() const { return traceInfo; }

    bool getTraceVerbose() const { return traceVerbose; }

    void flush() {
        if (traceToConsole) {
            std::cout << buffer;
        } else {
            std::ostringstream filename;
            std::string rankStr = rank >= 0 ? "_" + std::to_string(rank) : "";
            filename << outputDir << "/AsyncTrace_" << name << rankStr << ".log";

            std::ofstream outfile(filename.str(), std::ios::app);
            if (outfile.is_open()) {
                outfile << buffer;
                outfile.close();
            }
        }

        buffer.clear();
    }

    void addStackTrace() {
        if (traceInfo || traceVerbose) {
            std::ostringstream ss;
            ss << "Call stack:" << std::endl;

            const int max_frames = 64;
            void *frame_ptrs[max_frames];
            int num_frames = backtrace(frame_ptrs, max_frames);
            char **symbols = backtrace_symbols(frame_ptrs, num_frames);

            if (symbols == nullptr) {
                buffer += "Error retrieving backtrace symbols." + std::string("\n");
                return;
            }

            for (int i = 0; i < num_frames; ++i) {
                // Demangle the C++ function name
                size_t size;
                int status;
                char *demangled = abi::__cxa_demangle(symbols[i], nullptr, &size, &status);

                if (status == 0) {
                    ss << demangled << std::endl;
                    free(demangled);
                } else {
                    // Couldn't demangle, use the original symbol
                    ss << symbols[i] << std::endl;
                }
            }

            free(symbols);

            buffer += ss.str();

            if (buffer.size() >= bufferSize || nobuffer) {
                flush();
            }
        }
    }

private:
    void addTimestamp(std::ostringstream &stream) {
        auto now = std::chrono::system_clock::now();
        auto timePoint = std::chrono::system_clock::to_time_t(now);
        auto milliseconds =
                std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        std::tm tm;
#ifdef _WIN32
        localtime_s(&tm, &timePoint);
#else
        localtime_r(&timePoint, &tm);
#endif

        stream << "[" << tm.tm_year + 1900 << "/" << tm.tm_mon + 1 << "/" << tm.tm_mday << " " << tm.tm_hour
               << ":" << tm.tm_min << ":" << tm.tm_sec << ":" << milliseconds.count() << "] ";
    }

    std::string name;
    std::string buffer;
    size_t bufferSize;
    std::string outputDir;
    bool traceInfo = false;
    bool traceVerbose = false;
    bool traceToConsole = true;
    bool nobuffer = false;
    int rank = -1;
};

#define StartTraceInfo(logger) \
    if (logger.getTraceInfo()) logger.startTraceInfo()

#define TraceInfo(logger) \
    if (logger.getTraceInfo()) logger

#define PrintStack(logger) \
    if (logger.getTraceInfo()) logger.addStackTrace()

#define StartTraceVerbose(logger) \
    if (logger.getTraceVerbose()) logger.startTraceVerbose()

#define TraceVerbose(logger) \
    if (logger.getTraceVerbose()) logger

#endif  // ASYNC_LOGGER_HPP

/*
Example usage:

#include "logger.hpp"

int main() {
    AsyncLogger logger("CUD");

    StartTraceInfo(logger) << "This is an info message" << '\n';
    StartTraceVerbose(logger) << "This is a verbose message" << '\n';

    return 0;
}
*/
