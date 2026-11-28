// Enumerate at most three Sudoku completions. A node cap always marks a result uncertified.
#include <array>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
using Grid = std::array<int,81>;

struct Search {
    Grid grid{};
    int mask[27]{}, limit, nodes=0;
    bool reverse, capped=false;
    std::vector<Grid> found;
    static std::array<int,3> units(int i) {
        return {i/9,9+i%9,18+3*(i/27)+(i%9)/3};
    }
    int available(int i) const {
        auto u=units(i);
        return 1022 & ~(mask[u[0]] | mask[u[1]] | mask[u[2]]);
    }
    void toggle(int i, int d) {
        for (int u: units(i)) mask[u] ^= 1<<d;
    }
    void visit() {
        if (found.size()==3 || capped) return;
        if (++nodes > limit) { capped=true; return; }
        int best=-1, minimum=10, choices=0;
        for (int j=0;j<81;++j) {
            int i=reverse?80-j:j;
            if (grid[i]) continue;
            int bits=available(i), count=__builtin_popcount(static_cast<unsigned>(bits));
            if (!bits) return;
            if (count<minimum) {best=i; minimum=count; choices=bits;}
        }
        if (best<0) {found.push_back(grid);return;}
        for (int j=1;j<=9;++j) {
            int d=reverse?10-j:j;
            if (!(choices & (1<<d))) continue;
            grid[best]=d; toggle(best,d);
            visit();
            toggle(best,d); grid[best]=0;
            if (found.size()==3 || capped) break;
        }
    }
};

int main() {
    std::string line; int reverse, limit;
    while (std::cin>>line>>reverse>>limit) {
        if (line.size()!=81 || limit<=0) return 2;
        Search state{}; state.limit=limit; state.reverse=reverse;
        for (int i=0;i<81;++i) {
            int d=line[i]-'0';
            if (d<0 || d>9 || (d && !(state.available(i)&(1<<d)))) return 2;
            state.grid[i]=d;
            if (d) state.toggle(i,d);
        }
        state.visit();
        std::cout<<state.capped<<" "<<state.nodes<<" "<<state.found.size();
        for (const auto &g:state.found) {
            std::cout<<" "; for (int d:g) std::cout<<d;
        }
        std::cout<<std::endl;
    }
}
