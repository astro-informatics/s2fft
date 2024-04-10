#include <array>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>

int main() {
  thrust::default_random_engine rng(1337);
  thrust::uniform_int_distribution<int> dist;
  thrust::host_vector<int> h_vec(32 << 20);
  thrust::generate(h_vec.begin(), h_vec.end(), [&] { return dist(rng); });

  int nside = 4;
  int total_pixels = 12 * nside * nside;
  int number_of_rings = 2 * nside - 1;

  std::array<int, 23> ring_offsets;
  for (int i = 0; i < number_of_rings; i++)
    ring_offsets[i] = 12 * i * nside;

  std::cout << "nsides: " << nside << std::endl;
  std::cout << "total_pixels: " << total_pixels << std::endl;
  std::cout << "number_of_rings: " << number_of_rings << std::endl;
  std::cout << "ring_offsets: ";
  std::copy(ring_offsets.begin(), ring_offsets.end(),
            std::ostream_iterator<int>(std::cout, " "));

  // Allocate memory on cpu_data
}
