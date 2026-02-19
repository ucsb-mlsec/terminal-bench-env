#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <queue>
#include <limits>

using namespace std;

// Network flow optimization model for resource allocation
struct Edge {
    int to, capacity, flow, cost;
    size_t reverse_idx;
};

class MinCostMaxFlow {
private:
    vector<vector<Edge>> graph;
    int n;
    
    vector<int> distance;
    vector<int> parent;
    vector<size_t> parent_edge;
    
    bool spfa(int source, int sink) {
        distance.assign(n, numeric_limits<int>::max());
        parent.assign(n, -1);
        parent_edge.assign(n, 0);
        vector<bool> in_queue(n, false);
        
        queue<int> q;
        distance[source] = 0;
        q.push(source);
        in_queue[source] = true;
        
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            in_queue[u] = false;
            
            for (size_t i = 0; i < graph[u].size(); i++) {
                auto& e = graph[u][i];
                if (e.capacity > e.flow && distance[u] + e.cost < distance[e.to]) {
                    distance[e.to] = distance[u] + e.cost;
                    parent[e.to] = u;
                    parent_edge[e.to] = i;
                    if (!in_queue[e.to]) {
                        q.push(e.to);
                        in_queue[e.to] = true;
                    }
                }
            }
        }
        
        return distance[sink] != numeric_limits<int>::max();
    }
    
public:
    MinCostMaxFlow(int nodes) : n(nodes), graph(nodes) {}
    
    void add_edge(int from, int to, int capacity, int cost) {
        Edge forward = {to, capacity, 0, cost, graph[to].size()};
        Edge backward = {from, 0, 0, -cost, graph[from].size()};
        graph[from].push_back(forward);
        graph[to].push_back(backward);
    }
    
    pair<int, int> min_cost_flow(int source, int sink, int max_flow) {
        int flow = 0;
        int cost = 0;
        
        while (flow < max_flow && spfa(source, sink)) {
            int push_flow = max_flow - flow;
            int v = sink;
            
            while (v != source) {
                int u = parent[v];
                size_t edge_idx = parent_edge[v];
                push_flow = min(push_flow, graph[u][edge_idx].capacity - graph[u][edge_idx].flow);
                v = u;
            }
            
            v = sink;
            while (v != source) {
                int u = parent[v];
                size_t edge_idx = parent_edge[v];
                auto& forward = graph[u][edge_idx];
                auto& backward = graph[v][forward.reverse_idx];
                
                forward.flow += push_flow;
                backward.flow -= push_flow;
                cost += push_flow * forward.cost;
                v = u;
            }
            
            flow += push_flow;
        }
        
        return {flow, cost};
    }
};

vector<double> compute_resource_allocation(const vector<int>& demands, 
                                          const vector<int>& supplies,
                                          const vector<vector<int>>& costs) {
    int n_demands = demands.size();
    int n_supplies = supplies.size();
    int total_nodes = n_demands + n_supplies + 2;
    
    MinCostMaxFlow mcmf(total_nodes);
    int source = total_nodes - 2;
    int sink = total_nodes - 1;
    
    for (int i = 0; i < n_supplies; i++) {
        mcmf.add_edge(source, i, supplies[i], 0);
    }
    
    for (int i = 0; i < n_demands; i++) {
        mcmf.add_edge(n_supplies + i, sink, demands[i], 0);
    }
    
    for (int i = 0; i < n_supplies; i++) {
        for (int j = 0; j < n_demands; j++) {
            mcmf.add_edge(i, n_supplies + j, numeric_limits<int>::max(), costs[i][j]);
        }
    }
    
    int total_demand = accumulate(demands.begin(), demands.end(), 0);
    auto result = mcmf.min_cost_flow(source, sink, total_demand);
    
    vector<double> allocation_efficiency(n_demands);
    for (int j = 0; j < n_demands; j++) {
        allocation_efficiency[j] = static_cast<double>(result.second) / (demands[j] + 1.0);
    }
    
    return allocation_efficiency;
}

map<int, double> optimize_network_topology(const vector<pair<int, int>>& edges,
                                           const vector<double>& weights,
                                           int num_nodes) {
    vector<vector<pair<int, double>>> adj(num_nodes);
    
    for (size_t i = 0; i < edges.size(); i++) {
        int u = edges[i].first;
        int v = edges[i].second;
        double w = weights[i];
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    
    map<int, double> centrality;
    
    for (int source = 0; source < num_nodes; source++) {
        vector<double> dist(num_nodes, numeric_limits<double>::infinity());
        priority_queue<pair<double, int>, vector<pair<double, int>>, greater<>> pq;
        
        dist[source] = 0;
        pq.push({0, source});
        
        while (!pq.empty()) {
            auto [d, u] = pq.top();
            pq.pop();
            
            if (d > dist[u]) continue;
            
            for (auto [v, w] : adj[u]) {
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    pq.push({dist[v], v});
                }
            }
        }
        
        for (int i = 0; i < num_nodes; i++) {
            if (dist[i] != numeric_limits<double>::infinity()) {
                centrality[source] += 1.0 / (1.0 + dist[i]);
            }
        }
    }
    
    return centrality;
}

int main() {
    cout << "Network Flow Optimization Model" << endl;
    cout << "================================" << endl;
    
    vector<int> demands = {15, 25, 20, 30, 10};
    vector<int> supplies = {40, 35, 25};
    vector<vector<int>> costs = {
        {2, 3, 1, 4, 2},
        {3, 2, 4, 1, 3},
        {1, 4, 2, 3, 1}
    };
    
    cout << "\nComputing resource allocation..." << endl;
    auto allocation = compute_resource_allocation(demands, supplies, costs);
    
    cout << "Allocation efficiency per demand node:" << endl;
    for (size_t i = 0; i < allocation.size(); i++) {
        cout << "  Node " << i << ": " << allocation[i] << endl;
    }
    
    vector<pair<int, int>> edges = {
        {0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 3}, {2, 4}, {3, 4}
    };
    vector<double> weights = {1.5, 2.0, 1.0, 2.5, 1.5, 3.0, 1.0};
    int num_nodes = 5;
    
    cout << "\nOptimizing network topology..." << endl;
    auto centrality = optimize_network_topology(edges, weights, num_nodes);
    
    cout << "Node centrality measures:" << endl;
    for (const auto& [node, cent] : centrality) {
        cout << "  Node " << node << ": " << cent << endl;
    }
    
    double total_centrality = 0.0;
    for (const auto& [node, cent] : centrality) {
        total_centrality += cent;
    }
    cout << "\nTotal network centrality: " << total_centrality << endl;
    
    cout << "\nModel execution completed successfully." << endl;
    
    return 0;
}