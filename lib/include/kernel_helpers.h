// Adapted from code in a tutorial by Dan Foreman-Mackey
// https://github.com/dfm/extending-jax/blob/c33869665236877a2ae281/lib/kernel_helpers.h
//
// Original license:
//
// MIT License
//
// Copyright (c) 2021 Dan Foreman-Mackey
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// This header is not specific to our application and you'll probably want
// something like this for any extension you're building. This includes the
// infrastructure needed to serialize descriptors that are used with the
// "opaque" parameter of the GPU custom call. In our example we'll use this
// parameter to pass the size of our problem.

#ifndef _KERNEL_HELPERS_H_
#define _KERNEL_HELPERS_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace s2fft {

// https://en.cppreference.com/w/cpp/numeric/bit_cast
template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) &&
                            std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value,
                        To>::type
bit_cast(const From &src) noexcept {
  static_assert(std::is_trivially_constructible<To>::value,
                "This implementation additionally requires destination type to "
                "be trivially constructible");

  To dst;

  memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <typename T> std::string PackDescriptorAsString(const T &descriptor) {
  return std::string(bit_cast<const char *>(&descriptor), sizeof(T));
}

template <typename T>
const T *UnpackDescriptor(const char *opaque, std::size_t opaque_len) {
  if (opaque_len != sizeof(T)) {
    throw std::runtime_error("Invalid opaque object size");
  }
  return bit_cast<const T *>(opaque);
}

} // namespace s2fft

#endif // _KERNEL_HELPERS_H_
