#include <iostream>
#include <vector>
#include <unordered_set>
#include <pthread.h>
#include <limits>
#include <chrono>
#include <random>
#include <algorithm>
#include <atomic>
#include <iomanip>
#include <queue>        
#include <functional>   

using namespace std;

struct Edge {
    int from, to, weight;
};

static const long long INF = (long long)9e18;

// -----------------------------
// Sequential Bellman–Ford (64-bit)
// -----------------------------
vector<long long> bellman_ford_sequential(int n, const vector<Edge>& edges, int start) {
    vector<long long> dist(n, INF);
    dist[start] = 0;

    for (int it = 0; it < n - 1; ++it) {
        bool updated = false;
        for (const auto& e : edges) {
            if (dist[e.from] == INF) continue;
            long long cand = dist[e.from] + (long long)e.weight;
            if (cand < dist[e.to]) {
                dist[e.to] = cand;
                updated = true;
            }
        }
        if (!updated) break; // early exit
    }
    return dist;
}

// -----------------------------
// Sequential Dijkstra (binary heap). Requires non-negative weights.
// -----------------------------
vector<long long> dijkstra_sequential(int n, const vector<vector<pair<int,int>>>& adj, int start) {
    vector<long long> dist(n, INF);
    dist[start] = 0;

    using P = pair<long long,int>; // (dist, node)
    priority_queue<P, vector<P>, greater<P>> pq;
    pq.push({0, start});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto [v, w] : adj[u]) {
            long long nd = d + (long long)w;
            if (nd < dist[v]) {
                dist[v] = nd;
                pq.push({nd, v});
            }
        }
    }
    return dist;
}

// -----------------------------
// Parallel Bellman–Ford (pthreads + atomics)
// level-synchronous, double buffer with atomicMin on next[]
// -----------------------------
struct ThreadData {
    int thread_id;
    int start_idx, end_idx;
    const vector<Edge>* edges;
    const vector<long long>* dist;         // read-only current distances
    vector<atomic<long long>>* next;       // write-only (atomicMin) next distances
    atomic<bool>* any_updated;             // set true if any relaxation happens
};

inline bool atomic_min(std::atomic<long long>& ref, long long val) {
    long long cur = ref.load(std::memory_order_relaxed);
    while (val < cur) {
        if (ref.compare_exchange_weak(cur, val, std::memory_order_relaxed)) {
            return true; // updated
        }
        // cur reloaded; loop if still smaller
    }
    return false; // not updated
}

void* relax_edges(void* arg) {
    ThreadData* td = static_cast<ThreadData*>(arg);
    const auto& edges = *(td->edges);
    const auto& dist  = *(td->dist);
    auto& next        = *(td->next);
    auto& any_updated = *(td->any_updated);

    for (int i = td->start_idx; i < td->end_idx; ++i) {
        const Edge& e = edges[i];
        long long du = dist[e.from];
        if (du == INF) continue;
        long long cand = du + (long long)e.weight;
        if (atomic_min(next[e.to], cand)) {
            any_updated.store(true, std::memory_order_relaxed);
        }
    }
    return nullptr;
}

vector<long long> bellman_ford_parallel(int n, const vector<Edge>& edges, int start, int num_threads) {
    vector<long long> dist(n, INF);
    dist[start] = 0;

    // Pre-create thread containers
    vector<pthread_t> threads(num_threads);
    vector<ThreadData> tdata(num_threads);

    // Atomic next buffer (reused each iteration)
    vector<atomic<long long>> next(n);
    for (int i = 0; i < n; ++i) next[i].store(dist[i], std::memory_order_relaxed);

    const int m = (int)edges.size();
    const int chunk = (m + num_threads - 1) / num_threads;

    for (int it = 0; it < n - 1; ++it) {
        // next := dist
        for (int i = 0; i < n; ++i) next[i].store(dist[i], std::memory_order_relaxed);

        atomic<bool> any_updated(false);

        // Launch threads for this relaxation round
        for (int t = 0; t < num_threads; ++t) {
            int s = t * chunk;
            int e = std::min(s + chunk, m);
            tdata[t] = ThreadData{ t, s, e, &edges, &dist, &next, &any_updated };
            pthread_create(&threads[t], nullptr, relax_edges, &tdata[t]);
        }
        for (int t = 0; t < num_threads; ++t) pthread_join(threads[t], nullptr);

        // dist := next
        for (int i = 0; i < n; ++i) dist[i] = next[i].load(std::memory_order_relaxed);

        if (!any_updated.load(std::memory_order_relaxed)) break; // early exit
    }
    return dist;
}

// -----------------------------
// Random connected undirected graph generator
// avg_degree = desired average UNDIRECTED degree
// Produces a directed edge list (both directions for each undirected edge)
// -----------------------------
vector<Edge> generate_random_graph(int num_nodes, int avg_degree = 8, int max_weight = 10, uint32_t seed = 42) {
    mt19937 gen(seed);
    uniform_int_distribution<> weight_dist(1, max_weight);
    uniform_int_distribution<> node_dist(0, num_nodes - 1);

    vector<unordered_set<int>> nbr(num_nodes);
    vector<Edge> edges;
    edges.reserve(static_cast<size_t>(num_nodes) * avg_degree); // directed edges target

    // Ensure connectivity with a random spanning tree
    for (int v = 1; v < num_nodes; ++v) {
        uniform_int_distribution<> parent_dist(0, v - 1);
        int u = parent_dist(gen);
        int w = weight_dist(gen);
        nbr[u].insert(v);
        nbr[v].insert(u);
        edges.push_back({u, v, w});
        edges.push_back({v, u, w});
    }

    // Add extra undirected edges until directed edges reach target
    const int target_directed = max(2 * (num_nodes - 1), num_nodes * avg_degree); // at least the tree
    while ((int)edges.size() < target_directed) {
        int a = node_dist(gen), b = node_dist(gen);
        if (a == b || nbr[a].count(b)) continue;
        int w = weight_dist(gen);
        nbr[a].insert(b);
        nbr[b].insert(a);
        edges.push_back({a, b, w});
        edges.push_back({b, a, w});
    }

    return edges;
}

// Build adjacency list from directed edge list
static inline vector<vector<pair<int,int>>> build_adj(int n, const vector<Edge>& edges) {
    vector<vector<pair<int,int>>> adj(n);
    for (const auto& e : edges) adj[e.from].push_back({e.to, e.weight});
    return adj;
}

// Output: print the distances 
static void print_dist(const vector<long long>& d, const char* label) {
    cout << label << ": [";
    for (size_t i = 0; i < d.size(); ++i) {
        if (i) cout << ", ";
        if (d[i] >= INF/2) cout << "INF";   // unreachable
        else cout << d[i];
    }
    cout << "]\n";
}

// -----------------------------
// Main: timings & correctness check
// -----------------------------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Args: n avg_degree threads start max_weight seed
    int n           = (argc > 1) ? atoi(argv[1]) : 1000;
    int avg_degree  = (argc > 2) ? atoi(argv[2]) : 8;
    int threads     = (argc > 3) ? atoi(argv[3]) : 8;
    int start       = (argc > 4) ? atoi(argv[4]) : 0;
    int max_weight  = (argc > 5) ? atoi(argv[5]) : 10;
    uint32_t seed   = (argc > 6) ? (uint32_t)strtoul(argv[6], nullptr, 10) : 42u;

    cout << "Generating random graph...\n";
    cout << "nodes=" << n
         << " avg_degree=" << avg_degree
         << " threads=" << threads
         << " start=" << start
         << " max_weight=" << max_weight << "\n";

    auto edges = generate_random_graph(n, avg_degree, max_weight, seed);
    cout << "Generated " << edges.size() << " directed edges:\n";
    for (const auto &e : edges) {
        cout << e.from << " -> " << e.to << " (weight=" << e.weight << ")\n";
    }
    
    /*
    n = 4;            // A=0, B=1, C=2, D=3
    start = 0;        // source = A
    vector<Edge> edges = {
        {0,1,1}, {1,0,1},   // A<->B (1)
        {0,2,4}, {2,0,4},   // A<->C (4)
        {1,2,2}, {2,1,2},   // B<->C (2)
        {1,3,5}, {3,1,5},   // B<->D (5)
        {2,3,1}, {3,2,1}    // C<->D (1)
    };*/

    // Build adjacency for Dijkstra
    auto adj = build_adj(n, edges);

    // Sequential Dijkstra
    auto td1 = chrono::high_resolution_clock::now();
    auto dist_djk = dijkstra_sequential(n, adj, start);
    auto td2 = chrono::high_resolution_clock::now();
    double secs_djk = chrono::duration<double>(td2 - td1).count();
    cout << fixed << setprecision(6);
    cout << "Sequential Dijkstra:      " << secs_djk << " s\n";
    print_dist(dist_djk, "Output:");

    // Sequential Bellman-Ford
    auto t1 = chrono::high_resolution_clock::now();
    auto dist_seq = bellman_ford_sequential(n, edges, start);
    auto t2 = chrono::high_resolution_clock::now();
    double secs_seq = chrono::duration<double>(t2 - t1).count();
    cout << "Sequential Bellman-Ford:  " << secs_seq << " s\n";
    print_dist(dist_seq, "Output:");

    // Parallel Bellman-Ford
    auto t3 = chrono::high_resolution_clock::now();
    auto dist_par = bellman_ford_parallel(n, edges, start, threads);
    auto t4 = chrono::high_resolution_clock::now();
    double secs_par = chrono::duration<double>(t4 - t3).count();
    cout << "Parallel Bellman-Ford  (" << threads << " threads): " << secs_par << " s\n";
    print_dist(dist_par, "Output:");

    // Verify correctness
    size_t mism_djk_seq = 0, mism_seq_par = 0, mism_djk_par = 0;
    for (int i = 0; i < n; ++i) {
        if (dist_djk[i] != dist_seq[i]) ++mism_djk_seq;
        if (dist_seq[i] != dist_par[i]) ++mism_seq_par;
        if (dist_djk[i] != dist_par[i]) ++mism_djk_par;
    }
    cout << "Dijkstra vs Seq BF mismatches: " << mism_djk_seq << "\n";
    cout << "Seq BF vs Par BF mismatches:   " << mism_seq_par << "\n";
    cout << "Dijkstra vs Par BF mismatches: " << mism_djk_par << "\n";
    if (mism_djk_seq + mism_seq_par + mism_djk_par == 0) cout << "OK: all distances match.\n";

    return 0;
}
