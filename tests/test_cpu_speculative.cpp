// MIT. Past-only proposal, first-max ties and the exact recent-100 penalty.
#include "cpu_speculative.hpp"
#include <cstdio>
#include <stdexcept>

int main() {
    try {
        const auto require=[](bool ok){if(!ok)throw std::runtime_error("speculative policy control");};
        require(mt::cpu_ngram_draft({1,2,3,4,1,2},4)==std::vector<int32_t>({3,4,1,2}));
        require(mt::cpu_ngram_draft({1,2,3,4,1,2},2)==std::vector<int32_t>({3,4}));
        require(mt::cpu_ngram_draft({1,2,3,1,2,9,1,2},4)==std::vector<int32_t>({9,1,2}));
        require(mt::cpu_ngram_draft({1,2,3,4,5},4).empty());
        require(mt::cpu_ngram_draft({1,2},4).empty());
        require(mt::cpu_ngram_draft({1,2,1,2},0).empty());
        require(mt::cpu_greedy_choice({2,2,1},{},1.f)==0);
        require(mt::cpu_greedy_choice({2,1.9f},{0,0},1.1f)==1);
        require(mt::cpu_greedy_choice({-2,-2.1f},{0,0},1.1f)==1);
        require(mt::cpu_greedy_choice({2,1.9f},{0},1.f)==0);
        std::vector<int32_t> old(101,2);old[0]=0;
        require(mt::cpu_greedy_choice({2,1.9f},old,1.1f)==0);
        old[1]=0;require(mt::cpu_greedy_choice({2,1.9f},old,1.1f)==1);
        std::puts("{\"record\":\"speculativePolicy\",\"controls\":12,\"passed\":true}");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
