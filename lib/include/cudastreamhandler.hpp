
/**
 * @file cudastreamhandler.hpp
 * @brief Singleton class for managing CUDA streams and events.
 *
 * This header provides a singleton implementation that encapsulates the creation,
 * management, and cleanup of CUDA streams and events. It offers functions to fork
 * streams, add new streams, and synchronize (join) streams with a given dependency.
 *
 * Usage example:
 * @code
 *   #include "cudastreamhandler.hpp"
 *
 *   int main() {
 *       // Create a handler instance
 *       CudaStreamHandler handler;
 *
 *       // Fork 4 streams dependent on a given stream 'stream_main'
 *       handler.Fork(stream_main, 4);
 *
 *       // Do work on the forked streams...
 *
 *       // Join the streams back to 'stream_main'
 *       handler.join(stream_main);
 *
 *       return 0;
 *   }
 * @endcode
 *
 * Author: Wassim KABALAN
 */

#ifndef CUDASTREAMHANDLER_HPP
#define CUDASTREAMHANDLER_HPP

#include <algorithm>
#include <atomic>
#include <cuda_runtime.h>
#include <stdexcept>
#include <thread>
#include <vector>

// Singleton class managing CUDA streams and events
class CudaStreamHandlerImpl {
public:
    static CudaStreamHandlerImpl &instance() {
        static CudaStreamHandlerImpl instance;
        return instance;
    }

    void AddStreams(int numStreams) {
        if (numStreams > m_streams.size()) {
            int streamsToAdd = numStreams - m_streams.size();
            m_streams.resize(numStreams);
            std::generate(m_streams.end() - streamsToAdd, m_streams.end(), []() {
                cudaStream_t stream;
                cudaStreamCreate(&stream);
                return stream;
            });
        }
    }

    void join(cudaStream_t finalStream) {
        std::for_each(m_streams.begin(), m_streams.end(), [this, finalStream](cudaStream_t stream) {
            cudaEvent_t event;
            cudaEventCreate(&event);
            cudaEventRecord(event, stream);
            cudaStreamWaitEvent(finalStream, event, 0);
            m_events.push_back(event);
        });

        if (!cleanup_thread.joinable()) {
            stop_thread.store(false);
            cleanup_thread = std::thread([this]() { this->AsyncEventCleanup(); });
        }
    }

    // Fork function to add streams and set dependency on a given stream
    void Fork(cudaStream_t dependentStream, int N) {
        AddStreams(N);  // Add N streams

        // Set dependency on the provided stream
        std::for_each(m_streams.end() - N, m_streams.end(), [this, dependentStream](cudaStream_t stream) {
            cudaEvent_t event;
            cudaEventCreate(&event);
            cudaEventRecord(event, dependentStream);
            cudaStreamWaitEvent(stream, event, 0);  // Set the stream to wait on the event
            m_events.push_back(event);
        });
    }

    auto getIterator() { return StreamIterator(m_streams.begin(), m_streams.end()); }

    ~CudaStreamHandlerImpl() {
        stop_thread.store(true);
        if (cleanup_thread.joinable()) {
            cleanup_thread.join();
        }

        std::for_each(m_streams.begin(), m_streams.end(), cudaStreamDestroy);
        std::for_each(m_events.begin(), m_events.end(), cudaEventDestroy);
    }

    // Custom Iterator class to iterate over streams
    class StreamIterator {
    public:
        StreamIterator(std::vector<cudaStream_t>::iterator begin, std::vector<cudaStream_t>::iterator end)
                : current(begin), end(end) {}

        cudaStream_t next() {
            if (current == end) {
                throw std::out_of_range("No more streams.");
            }
            return *current++;
        }

        bool hasNext() const { return current != end; }

    private:
        std::vector<cudaStream_t>::iterator current;
        std::vector<cudaStream_t>::iterator end;
    };

private:
    CudaStreamHandlerImpl() : stop_thread(false) {}
    CudaStreamHandlerImpl(const CudaStreamHandlerImpl &) = delete;
    CudaStreamHandlerImpl &operator=(const CudaStreamHandlerImpl &) = delete;

    void AsyncEventCleanup() {
        while (!stop_thread.load()) {
            std::for_each(m_events.begin(), m_events.end(), [this](cudaEvent_t &event) {
                if (cudaEventQuery(event) == cudaSuccess) {
                    cudaEventDestroy(event);
                    event = nullptr;
                }
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    std::vector<cudaStream_t> m_streams;
    std::vector<cudaEvent_t> m_events;
    std::thread cleanup_thread;
    std::atomic<bool> stop_thread;
};

// Public class for encapsulating the singleton operations
class CudaStreamHandler {
public:
    CudaStreamHandler() = default;
    ~CudaStreamHandler() = default;

    void AddStreams(int numStreams) { CudaStreamHandlerImpl::instance().AddStreams(numStreams); }

    void join(cudaStream_t finalStream) { CudaStreamHandlerImpl::instance().join(finalStream); }

    void Fork(cudaStream_t cudastream, int N) { CudaStreamHandlerImpl::instance().Fork(cudastream, N); }

    // Get the custom iterator for CUDA streams
    CudaStreamHandlerImpl::StreamIterator getIterator() {
        return CudaStreamHandlerImpl::instance().getIterator();
    }
};

#endif  // CUDASTREAMHANDLER_HPP
