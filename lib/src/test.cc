#include <algorithm>
#include <iostream>
#include <ranges>

int main() {
    int nside = 4;
    int nphi = 4;
    int L = 2 * nside;
    int ftm_size = 2 * L;

    auto set = std::ranges::iota_view{0, nphi};
    auto res_seq = std::ranges::iota_view{0, ftm_size};

    for (auto i : res_seq) {
        int indx = (L - nphi / 2 - i);
        std::cout << "Previous [" << i << "] new [" << indx << "]" << std::endl;
    }
}