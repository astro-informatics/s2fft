// Adapted from code by JAX authors
// https://github.com/jax-ml/jax/blob/3d389a7fb440c412d/jaxlib/kernel_nanobind_helpers.h

/* Copyright 2019 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef _KERNEL_NANOBIND_HELPERS_H_
#define _KERNEL_NANOBIND_HELPERS_H_

#include <string>

#include "nanobind/nanobind.h"
#include "kernel_helpers.h"

namespace s2fft {

// Descriptor objects are opaque host-side objects used to pass data from JAX
// to the custom kernel launched by XLA. Currently simply treat host-side
// structures as byte-strings; this is not portable across architectures. If
// portability is needed, we could switch to using a representation such as
// protocol buffers or flatbuffers.

// Packs a descriptor object into a nanobind::bytes structure.
// UnpackDescriptor() is available in kernel_helpers.h.
template <typename T>
nanobind::bytes PackDescriptor(const T& descriptor) {
  std::string s = PackDescriptorAsString(descriptor);
  return nanobind::bytes(s.data(), s.size());
}

template <typename T>
nanobind::capsule EncapsulateFunction(T* fn) {
  return nanobind::capsule(bit_cast<void*>(fn),
                           "xla._CUSTOM_CALL_TARGET");
}

}  // namespace s2fft 

#endif  // _KERNEL_NANOBIND_HELPERS_H_
