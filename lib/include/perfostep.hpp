
/**
 * @file Perfostep.hpp
 * @version 0.0.4
 * @brief A class for measuring the performance of code execution.
 * @details This header-only C++ class provides functionality for measuring the
 * performance of code execution, supporting both CPU and GPU measurements. It
 * utilizes high-resolution timers and optional NVIDIA Tools Extension (NVTX)
 * for GPU measurements.
 *
 * Environment Variable:
 * - ENABLE_PERFO_STEP: Set this environment variable to enable performance
 * measurement. Possible values are "TIMER" for CPU timing or "NVTX" for NSYS
 * profiling or "CUDA" for CUDA timing.
 * @code
 * // Example 1: Measure the performance of a CPU code
 * Perfostep perf;
 * perf.start("Code block 1");
 * // Code block to measure
 * for (int i = 0; i < 1000000; ++i) {
 *   // Some computation
 * }
 * perf.stop();
 *
 * perf.printToMarkdown("perf_report.md");
 * perf.printToCSV("perf_report.csv");
 *
 * // Example 2: Measure the performance of GPU code
 * Perfostep perf;
 * perf.start("CUDA Kernel");
 * // Kernel to measure
 * myCUDAKernel<<<blocks, threads>>>(input, output);
 * perf.stop();
 *
 * perf.printToMarkdown("cuperf_report.md");
 * perf.printToCSV("cuperf_report.csv");
 * @endcode
 *
 *
 * @author Wassim KABALAN
 */

#ifndef PERFOSTEP_HPP
#define PERFOSTEP_HPP

#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#if __has_include(<nvToolsExt.h>)
#define ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define ENABLE_CUDA
#endif

using namespace std::chrono;

typedef std::map<const std::string, const std::string> ColumnNames;
typedef std::map<const std::string, double> Reports;

class AbstractPerfostep {
public:
    virtual void Start(const std::string &iReport, const ColumnNames &iCol) = 0;

    virtual double Stop() = 0;
    virtual void Report(const bool &iPrintTotal = false) const = 0;
    virtual void PrintToMarkdown(const char *ifilename, const bool &iPrintTotal = false) const = 0;
    virtual void PrintToCSV(const char *ifilename, const bool &iPrintTotal = false) const = 0;
    virtual void Switch(const std::string &iReport, const ColumnNames &iCol) = 0;
    virtual ~AbstractPerfostep() {}

protected:
    Reports m_Reports; /**< The report of measured tasks. */
    ColumnNames m_ColNames;
};

class BasePerfostep : public AbstractPerfostep {
public:
    void Report(const bool &iPrintTotal = false) const override {
        if (m_Reports.size() == 0) return;

        std::cout << "Reporting : " << std::endl;
        std::cout << "For parameters: " << std::endl;

        for (const auto &entry : m_ColNames) {
            std::cout << std::get<0>(entry) << " : " << std::get<1>(entry) << std::endl;
        }

        for (const auto &entry : m_Reports) {
            std::cout << std::get<0>(entry) << " : " << std::get<1>(entry) << "ms " << std::endl;
        }
    }

    void PrintToMarkdown(const char *filename, const bool &iPrintTotal = false) const override {
        if (m_Reports.size() == 0) return;

        std::ofstream file(filename, std::ios::app);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + std::string(filename));
        }

        if (file.tellp() == 0) {  // Check if file is empty
            file << "| Task | ";
            for (const auto &entry : m_ColNames) {
                file << std::get<0>(entry) << " | ";
            }
            file << "Elapsed Time (ms) |" << std::endl;  // Header names for columns
            // For the Task column
            file << "| --- | ";
            // For the other columns
            for (const auto &entry : m_ColNames) {
                file << " --- | ";
            }
            // For the elapsed time column
            file << " --------------- |" << std::endl;
        }
        std::string colvalues;
        for (const auto &col : m_ColNames) {
            colvalues += std::get<1>(col) + " | ";
        }

        for (const auto &entry : m_Reports) {
            file << "| " << std::get<0>(entry) << " | " << colvalues << std::get<1>(entry) << " |"
                 << std::endl;
        }

        if (iPrintTotal) file << "| Total | " << colvalues << GetTotal() << " |" << std::endl;
        file.close();
    }
    /**
     * @brief Prints the measured tasks and their elapsed times in a CSV
     * format to a file.
     * @param filename The name of the file to write the CSV data to.
     */
    void PrintToCSV(const char *filename, const bool &iPrintTotal) const override {
        if (m_Reports.size() == 0) return;

        std::ofstream file(filename, std::ios::app);  // Open file in append mode

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + std::string(filename));
        }

        if (file.tellp() == 0) {  // Check if file is empty
            file << "Task,";
            for (const auto &entry : m_ColNames) {
                file << std::get<0>(entry) << ",";
            }
            file << "Elapsed Time (ms)" << std::endl;  // Header names for columns
        }

        std::string colvalues;
        for (const auto &col : m_ColNames) {
            colvalues += std::get<1>(col) + ",";
        }

        for (const auto &entry : m_Reports) {
            file << std::get<0>(entry) << "," << colvalues << std::get<1>(entry) << std::endl;
        }

        if (iPrintTotal) file << "Total," << colvalues << GetTotal() << std::endl;
        file.close();
    }

    void Switch(const std::string &iReport, const ColumnNames &iCol) override {
        Stop();
        Start(iReport, iCol);
    }

private:
    double GetTotal() const {
        double total = std::accumulate(m_Reports.begin(), m_Reports.end(), 0.0,
                                       [](double sum, const std::tuple<const std::string, double> &entry) {
                                           return sum + std::get<1>(entry);
                                       });
        return total;
    }
};

typedef std::vector<std::tuple<const std::string, time_point<high_resolution_clock>>> StartTimes;

class PerfostepChrono : public BasePerfostep {
public:
    void Start(const std::string &iReport, const ColumnNames &iCol) override {
        m_StartTimes.push_back(std::make_tuple(iReport, high_resolution_clock::now()));
        m_ColNames = iCol;
    }

    double Stop() override {
        // Check if there are any start times
        assert(m_StartTimes.size() > 0);

        m_EndTime = high_resolution_clock::now();
        duration<double> diff = m_EndTime - std::get<1>(m_StartTimes.back());
        double elapsed_time = diff.count() * 1000;
        m_Reports[std::get<0>(m_StartTimes.back())] = elapsed_time;
        m_StartTimes.pop_back();
        return elapsed_time;
    }

    ~PerfostepChrono() {
        if (m_StartTimes.size() > 0) {
            std::cerr << "Warning: There are still start times not stopped" << std::endl;
            // print message for each start time
            for (const auto &entry : m_StartTimes) {
                std::cerr << "Start time for " << std::get<0>(entry) << " is not stopped" << std::endl;
            }
        }
    }

private:
    StartTimes m_StartTimes;                     /**< The start time of the measurement. */
    time_point<high_resolution_clock> m_EndTime; /**< The end time of the measurement. */
};

#ifdef ENABLE_NVTX

// Credit : https://github.com/NVIDIA/cuDecomp
class PerfostepNVTX : public BasePerfostep {
public:
    // ColumnNames are not used in NVTX
    void Start(const std::string &iReport, const ColumnNames &iCol) override {
        static constexpr int ncolors_ = 8;
        static constexpr int colors_[ncolors_] = {0x3366CC, 0xDC3912, 0xFF9900, 0x109618,
                                                  0x990099, 0x3B3EAC, 0x0099C6, 0xDD4477};
        std::string range_name(iReport);
        std::hash<std::string> hash_fn;
        int color = colors_[hash_fn(range_name) % ncolors_];
        nvtxEventAttributes_t ev = {0};
        ev.version = NVTX_VERSION;
        ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        ev.colorType = NVTX_COLOR_ARGB;
        ev.color = color;
        ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
        ev.message.ascii = range_name.c_str();
        nvtxRangePushEx(&ev);
        nvtx_ranges++;
    }

    double Stop() override {
        nvtxRangePop();
        nvtx_ranges--;
        assert(nvtx_ranges >= 0);
        return 0.0;
    }
    ~PerfostepNVTX() {
        if (nvtx_ranges > 0) {
            std::cerr << "Warning: There are still start times not stopped" << std::endl;
            for (int i = 0; i < nvtx_ranges; i++) nvtxRangePop();
        }
    }

private:
    int nvtx_ranges = 0;
};

#endif  // ENABLE_NVTX

#ifdef ENABLE_CUDA

typedef std::vector<std::tuple<const std::string, cudaEvent_t>> StartEvents;

class PerfostepCUDA : public BasePerfostep {
public:
    PerfostepCUDA() { cudaEventCreate(&m_EndEvent); }

    void Start(const std::string &iReport, const ColumnNames &iCol) override {
        cudaEvent_t m_StartEvent;
        cudaEventCreate(&m_StartEvent);
        cudaEventRecord(m_StartEvent);
        m_StartEvents.push_back(std::make_tuple(iReport, m_StartEvent));
        m_ColNames = iCol;
    }

    double Stop() override {
        cudaEventRecord(m_EndEvent);
        cudaEventSynchronize(m_EndEvent);
        float elapsed;
        cudaEventElapsedTime(&elapsed, std::get<1>(m_StartEvents.back()), m_EndEvent);
        double m_ElapsedTime = static_cast<double>(elapsed);
        cudaEventDestroy(std::get<1>(m_StartEvents.back()));
        m_Reports[std::get<0>(m_StartEvents.back())] = m_ElapsedTime;
        m_StartEvents.pop_back();

        return m_ElapsedTime;
    }

    ~PerfostepCUDA() {
        if (m_StartEvents.size() > 0) {
            std::cerr << "Warning: There are still start events not stopped" << std::endl;
            std::for_each(m_StartEvents.cbegin(), m_StartEvents.cend(),
                          [](const std::tuple<const std::string, cudaEvent_t> &entry) {
                              cudaEventDestroy(std::get<1>(entry));
                          });
        }
        cudaEventDestroy(m_EndEvent);
    }

private:
    StartEvents m_StartEvents; /**< The start event for CUDA measurement. */
    cudaEvent_t m_EndEvent;    /**< The end event for CUDA measurement. */
};

#endif  // ENABLE_CUDA

class Perfostep {
public:
    Perfostep() {
        static const char *env = std::getenv("ENABLE_PERFO_STEP");
        if (env != nullptr) {
            std::string envStr(env);
            if (envStr == "TIMER") {
                m_Perfostep = std::make_unique<PerfostepChrono>();
                m_EnablePerfoStep = true;
            } else if (envStr == "NVTX") {
#ifdef ENABLE_NVTX
                m_Perfostep = std::make_unique<PerfostepNVTX>();
                m_EnablePerfoStep = true;
#else
                throw std::runtime_error("NVTX is not available. Please install NVTX to use it.");
#endif
            } else if (envStr == "CUDA") {
#ifdef ENABLE_CUDA
                m_Perfostep = std::make_unique<PerfostepCUDA>();
                m_EnablePerfoStep = true;
#else
                throw std::runtime_error(
                        "CUDA is not available. Please install CUDA "
                        "or compile using nvcc to use it.");
#endif
            } else {
                throw std::runtime_error("Invalid value for ENABLE_PERFO_STEP: " + envStr +
                                         ". Possible values are TIMER, NVTX, or "
                                         "CUDA.");
            }
        }
    }

    void Start(const std::string &iReport, const ColumnNames &iCol = {}) {
        if (m_EnablePerfoStep) m_Perfostep->Start(iReport, iCol);
    }
    double Stop() {
        if (m_EnablePerfoStep) return m_Perfostep->Stop();
        return 0.0;
    }
    void Report(const bool &iPrintTotal = false) const {
        if (m_EnablePerfoStep) m_Perfostep->Report(iPrintTotal);
    }
    void PrintToMarkdown(const char *filename, const bool &iPrintTotal = false) const {
        if (m_EnablePerfoStep) m_Perfostep->PrintToMarkdown(filename, iPrintTotal);
    }
    void PrintToCSV(const char *filename, const bool &iPrintTotal = false) const {
        if (m_EnablePerfoStep) m_Perfostep->PrintToCSV(filename, iPrintTotal);
    }
    void Switch(const std::string &iReport, const ColumnNames &iCol = {}) {
        if (m_EnablePerfoStep) m_Perfostep->Switch(iReport, iCol);
    }

private:
    std::unique_ptr<AbstractPerfostep> m_Perfostep;
    bool m_EnablePerfoStep = false;
};
#endif  // PERFOSTEP_HPP
