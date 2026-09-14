/**
 * @file cudastreamhandler.hpp
 * @brief Singleton class for managing per-device CUDA streams and events.
 *
 * Streams and events are device-scoped CUDA objects: a stream created while device 0 is
 * current cannot be used from another device. The pool is therefore keyed by device ordinal
 * (single-process multi-GPU, astro-informatics/s2fft#389).
 *
 * Events are one-shot dependencies: cudaStreamWaitEvent captures the event's recorded state
 * at enqueue time, so each event is destroyed as soon as its wait is enqueued. A deferred
 * cleanup thread (and its lock) is not needed and would contend with the transform handler.
 *
 * Usage example:
 * @code
 *   #include "cudastreamhandler.hpp"
 *
 *   int main() {
 *       // Create a handler instance
 *       CudaStreamHandler handler;
 *       // Fork 4 streams on device 0, dependent on stream 'main'
 *       auto forked = handler.Fork(0, main, 4);
 *       // Do work on the forked streams...
 *       // Join them back into 'main'
 *       handler.join(main, forked);
 *       return 0;
 *   }
 * @endcode
 *
 * Author: Wassim KABALAN
 */

#ifndef CUDASTREAMHANDLER_HPP
#define CUDASTREAMHANDLER_HPP

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "cudadeviceguard.hpp"

// Singleton class managing per-device CUDA streams and events
class CudaStreamHandlerImpl {
public:
    static CudaStreamHandlerImpl &instance() {
        static CudaStreamHandlerImpl instance;
        return instance;
    }

    /**
     * @brief Forks N streams on `device`, each waiting on the current state of `dependentStream`.
     *
     * Returns exactly the N streams to use for this fork, so a later Fork with a smaller N can
     * never hand out streams whose dependencies were set for a different batch count.
     */
    std::vector<cudaStream_t> Fork(int device, cudaStream_t dependentStream, int N) {
        if (N <= 0) {
            throw std::invalid_argument("Fork requires at least one stream.");
        }
        std::vector<cudaStream_t> &pool = m_streams_by_device[device];
        if (pool.size() < static_cast<size_t>(N)) {
            CudaDeviceScope guard(device);
            while (pool.size() < static_cast<size_t>(N)) {
                cudaStream_t stream;
                if (cudaStreamCreate(&stream) != cudaSuccess) {
                    throw std::runtime_error("Failed to create a CUDA stream.");
                }
                pool.push_back(stream);
            }
        }

        std::vector<cudaStream_t> forked(pool.end() - N, pool.end());
        std::for_each(forked.begin(), forked.end(), [dependentStream](cudaStream_t stream) {
            cudaEvent_t event;
            cudaEventCreate(&event);
            cudaEventRecord(event, dependentStream);
            cudaStreamWaitEvent(stream, event, 0);  // Set the stream to wait on the event
            cudaEventDestroy(event);
        });
        return forked;
    }

    /**
     * @brief Joins the forked streams back into `finalStream` (each waits on their completion).
     */
    void join(cudaStream_t finalStream, const std::vector<cudaStream_t> &forked) {
        std::for_each(forked.begin(), forked.end(), [finalStream](cudaStream_t stream) {
            cudaEvent_t event;
            cudaEventCreate(&event);
            cudaEventRecord(event, stream);
            cudaStreamWaitEvent(finalStream, event, 0);
            cudaEventDestroy(event);
        });
    }

    ~CudaStreamHandlerImpl() {
        for (auto &entry : m_streams_by_device) {
            std::for_each(entry.second.begin(), entry.second.end(), cudaStreamDestroy);
        }
    }

private:
    CudaStreamHandlerImpl() = default;
    CudaStreamHandlerImpl(const CudaStreamHandlerImpl &) = delete;
    CudaStreamHandlerImpl &operator=(const CudaStreamHandlerImpl &) = delete;

    // Streams keyed by device ordinal: stream creation binds them to the device that is
    // current at creation time.
    std::unordered_map<int, std::vector<cudaStream_t>> m_streams_by_device;
};

// Public class for encapsulating the singleton operations
class CudaStreamHandler {
public:
    CudaStreamHandler() = default;
    ~CudaStreamHandler() = default;

    std::vector<cudaStream_t> Fork(int device, cudaStream_t cudastream, int N) {
        return CudaStreamHandlerImpl::instance().Fork(device, cudastream, N);
    }

    void join(cudaStream_t cudastream, const std::vector<cudaStream_t> &forked) {
        CudaStreamHandlerImpl::instance().join(cudastream, forked);
    }
};

#endif  // CUDASTREAMHANDLER_HPP
