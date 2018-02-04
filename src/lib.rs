// Enumerate shortest s-t paths in a graph, following the algorithm in
// http://citeseer.ist.psu.edu/viewdoc/download;jsessionid=7BB56B2ABC7C9113C121413A62AF3974?doi=10.1.1.30.3705&rep=rep1&type=pdf
// For sake of simplicity, when enumerating the paths we simply do
// a Dijkstra-esque walk with an auxilliary heap, so the i'th path
// is generated in O(log i), rather than constant time. This also means
// that no explicit path count or length bound needs to be given, and
// paths are generated in increasing order of length.
extern crate petgraph;

mod scored;

#[cfg(test)]
mod tests;

use petgraph::{Direction, Graph};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::algo::Measure;
use petgraph::visit::{EdgeRef, IntoEdgesDirected, Visitable, VisitMap};
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::ops::Sub;

use scored::{MinScored, Scored};

// Compute the shortest path tree toward the target. Returns it as a map
// mapping node IDs to the distance from the target and the outgoing edge
// in the SPT.
//
// The code is similar to that of the `dijkstra` function in petgraph, but
// the main Dijkstra loop is structured slightly differently.
pub fn shortest_path_tree<G, F, K>(graph: G, target: G::NodeId, mut edge_cost: F)
    -> HashMap<G::NodeId, (K, Option<G::EdgeId>)> 
    where
    G: IntoEdgesDirected + Visitable,
    G::NodeId: Eq + Hash,
    F: FnMut(G::EdgeRef) -> K,
    K: Measure + Copy,
{
    let mut visited = graph.visit_map();
    let mut tree = HashMap::new();
    let mut visit_next = BinaryHeap::new();
    let zero_score = K::default();

    tree.insert(target, (zero_score, None));
    visit_next.push(MinScored(zero_score, target));

    while let Some(MinScored(node_score, node)) = visit_next.pop() {
        if visited.is_visited(&node) {
            continue
        }

        visited.visit(node);

        for e in graph.edges_directed(node, Direction::Incoming) {
            let next = e.source();
            if visited.is_visited(&next) {
                continue
            }

            visited.visit(node);

            let next_score = node_score + edge_cost(e);
            match tree.entry(next) {
                Entry::Occupied(ent) => {
                    let (previous_score, _) = *ent.get();
                    if next_score < previous_score {
                        *ent.into_mut() = (next_score, Some(e.id()));
                    } else {
                        continue
                    }
                },
                Entry::Vacant(ent) => {
                    ent.insert((next_score, Some(e.id())));
                },
            }

            visit_next.push(MinScored(next_score, next));
        }
    }

    tree
}

// Compute the graph denoted D(G) in the source paper, and the mapping
// V(G) -> V(D(G))
// A vertex can be missing from this mapping if the only outgoing paths
// toward the sink are in the shortest path tree.
pub fn heap_graph<N, E, K, F>(
    graph: &Graph<N, E>,
    target: NodeIndex,
    spt: &HashMap<NodeIndex, (K, Option<EdgeIndex>)>,
    mut edge_cost: F,
) -> (Graph<(EdgeIndex, K), ()>, HashMap<NodeIndex, NodeIndex>)
    where
    E: Copy,
    F: FnMut(E) -> K,
    K: Measure + Sub<Output=K> + Copy,
{
    let mut result_graph = Graph::<(EdgeIndex, K), ()>::new();
    let mut heap_roots = HashMap::new();

    let mut inverse_spt: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();
    for (&node, &(_, successor)) in spt.iter() {
        if let Some(successor_edge) = successor {
            let successor_node = graph.edge_endpoints(successor_edge).unwrap().1;
            inverse_spt.entry(successor_node).or_insert_with(Vec::new).push(node);
        }
    }

    let mut edge_sidetrack_cost = |e: EdgeIndex| -> K {
        let (edge_source, edge_target) = match graph.edge_endpoints(e) {
            Some(endpoints) => endpoints,
            None => return K::default(),
        };
        let source_distance = match spt.get(&edge_source) {
            Some(&(dist, _)) => dist,
            None => return K::default(),
        };
        let target_distance = match spt.get(&edge_target) {
            Some(&(dist, _)) => dist,
            None => return K::default(),
        };
        edge_cost(graph[e]) + target_distance - source_distance
    };

    let mut visit_order = VecDeque::new();
    visit_order.push_back(target);

    println!();
    while let Some(node) = visit_order.pop_front() {
        println!("{:?}", node);
        let (_, successor) = spt[&node];
        match successor {
            Some(successor_edge_id) => {
                let successor_target = graph.edge_endpoints(successor_edge_id).unwrap().1;

                match heap_roots.get(&successor_target) {
                    Some(&successor_heap_root) => {
                        let mut outgoing_edges: Vec<EdgeIndex> = Vec::new();
                        for e in graph.edges(node) {
                            let e_id = e.id();
                            if e_id == successor_edge_id {
                                continue
                            }
                            outgoing_edges.push(e_id);
                        }

                        let new_root = merge_and_embed(&mut result_graph, outgoing_edges, successor_heap_root, &mut edge_sidetrack_cost);
                        heap_roots.insert(node, new_root);
                    },
                    None => {
                        // The target heap is empty, so we simply embed the appropriate heap
                        let mut scored_edges: Vec<Scored<K, EdgeIndex>> = Vec::new();
                        for e in graph.edges(node) {
                            let e_id = e.id();
                            if e_id == successor_edge_id {
                                continue
                            }
                            scored_edges.push(Scored(edge_sidetrack_cost(e_id), e_id));
                        }

                        if scored_edges.len() > 0 {
                            min_heapify(&mut scored_edges);

                            let mut heap_nodes = Vec::new();
                            for &Scored(e_cost, e_id) in scored_edges.iter() {
                                let new_node = result_graph.add_node((e_id, e_cost));
                                heap_nodes.push(new_node);
                            }
                            for (i, &node) in heap_nodes.iter().enumerate() {
                                if 2*i + 1 < heap_nodes.len() {
                                    let child_node = heap_nodes[2*i+1];
                                    result_graph.add_edge(node, child_node, ());
                                }
                                if 2*i + 2 < heap_nodes.len() {
                                    let child_node = heap_nodes[2*i+2];
                                    result_graph.add_edge(node, child_node, ());
                                }
                            }
                            heap_roots.insert(node, heap_nodes[0]);
                        }
                    },
                }
            },
            None => {
                // This is the target vertex, so we simply embed a heap of all
                // outgoing edges.
                let mut scored_edges: Vec<Scored<K, EdgeIndex>> = Vec::new();
                for e in graph.edges(node) {
                    let e_id = e.id();
                    scored_edges.push(Scored(edge_sidetrack_cost(e_id), e_id));
                }
                if scored_edges.len() > 0 {
                    min_heapify(&mut scored_edges);

                    let mut heap_nodes = Vec::new();
                    for &Scored(e_cost, e_id) in scored_edges.iter() {
                        let new_node = result_graph.add_node((e_id, e_cost));
                        heap_nodes.push(new_node);
                    }
                    for (i, &node) in heap_nodes.iter().enumerate() {
                        if 2*i + 1 < heap_nodes.len() {
                            let child_node = heap_nodes[2*i+1];
                            result_graph.add_edge(node, child_node, ());
                        }
                        if 2*i + 2 < heap_nodes.len() {
                            let child_node = heap_nodes[2*i+2];
                            result_graph.add_edge(node, child_node, ());
                        }
                    }
                    heap_roots.insert(node, heap_nodes[0]);
                }
            },
        }
        if let Some(predecessors) = inverse_spt.get(&node) {
            for &pred in predecessors.iter() {
                visit_order.push_back(pred);
            }
        }
    }

    (result_graph, heap_roots)
}

// Creates a heap out of the given values, merges it with the existing heap
// at target_root, and returns the root of the new merged heap. The existing
// heap should be untouched.
fn merge_and_embed<N, K, F>(graph: &mut Graph<(N, K), ()>, values: Vec<N>, target_root: NodeIndex, mut cost: F) -> NodeIndex
    where
    N: Copy,
    F: FnMut(N) -> K,
    K: PartialOrd + Copy,
{
    // If there is nothing to merge, we can just use the existing heap root.
    if values.len() == 0 {
        return target_root
    }

    let mut scored_values = Vec::new();
    for v in values {
        scored_values.push(Scored(cost(v), v));
    }
    min_heapify(&mut scored_values);

    // Create a new 3-heap where the root has out-degree 1, including
    // the root of the 2-heap and the entire 3-heap.
    let new_root = make_root(graph, scored_values[0].1, target_root, cost);

    // Create nodes for the remaining nodes of the 2-heap and link them together.
    let mut heap_nodes = Vec::new();
    heap_nodes.push(new_root);

    for &Scored(v_cost, v) in scored_values.iter().skip(1) {
        let new_node = graph.add_node((v, v_cost));
        heap_nodes.push(new_node);
    }

    for (i, &node) in heap_nodes.iter().enumerate() {
        if 2*i + 1 < heap_nodes.len() {
            let child_node = heap_nodes[2*i + 1];
            graph.add_edge(node, child_node, ());
        }
        if 2*i + 2 < heap_nodes.len() {
            let child_node = heap_nodes[2*i + 2];
            graph.add_edge(node, child_node, ());
        }
    }

    new_root
}

// Conceptually create a node linking to the given root and sift down,
// but avoid making any unnecessary nodes. Returns the index of the new root,
// which will have out-degree 1 (so that it can be linked to two other nodes).
fn make_root<N, F, K>(graph: &mut Graph<(N, K), ()>, new_value: N, target_root: NodeIndex, mut cost: F) -> NodeIndex
    where
    N: Copy,
    F: FnMut(N) -> K,
    K: PartialOrd + Copy,
{
    let value_cost = cost(new_value);
    let root_cost = graph[target_root].1;
    if value_cost <= root_cost {
        let new_root = graph.add_node((new_value, value_cost));
        graph.add_edge(new_root, target_root, ());
        return new_root;
    }
    let mut least_child_edge = None;
    for e in graph.edges(target_root) {
        match least_child_edge {
            None => {
                let child_node = e.target();
                let child_cost = graph[child_node].1;
                least_child_edge = Some((child_cost, e.id()));
            },
            Some((old_child_cost, _)) => {
                let child_node = e.target();
                let child_cost = graph[child_node].1;
                if child_cost < old_child_cost {
                    least_child_edge = Some((child_cost, e.id()));
                }
            },
        }
    }
    match least_child_edge {
        Some((_, e_id)) => {
            let least_child_node = graph.edge_endpoints(e_id).unwrap().1;
            // Put the new value into the least child's subtree, which will
            // result in a root no larger than the smallest child.
            let new_child = make_root(graph, new_value, least_child_node, cost);
            // Link the resulting root to the other children of the old root.
            let mut targets = Vec::new();
            for e in graph.edges(target_root) {
                if e.id() == e_id {
                    continue
                }
                targets.push(e.target());
            }
            for t in targets {
                graph.add_edge(new_child, t, ());
            }
            // Create a new root with the value of the old root, and link it
            // to the child created above.
            let target_root_value = graph[target_root];
            let new_root = graph.add_node(target_root_value);
            graph.add_edge(new_root, new_child, ());

            new_root
        },
        None => {
            // The target root doesn't have any children, so we just link
            // the new nodes in reverse order.
            let new_child = graph.add_node((new_value, cost(new_value)));
            let (target_root_value, target_root_cost) = graph[target_root];
            let new_root = graph.add_node((target_root_value, target_root_cost));
            graph.add_edge(new_root, new_child, ());

            new_root
        },
    }
}

fn min_heapify<T: PartialOrd>(vec: &mut [T]) {
    let mut n = vec.len() / 2;
    while n > 0 {
        n -= 1;
        sift_down(vec, n);
    }
}

#[test]
fn test_heapify() {
    let mut vec = vec![3,1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2,3];
    min_heapify(&mut vec);

    for (i, &v) in vec.iter().enumerate() {
        if 2*i + 1 < vec.len() {
            assert!(v <= vec[2*i + 1]);
        }
        if 2*i + 2 < vec.len() {
            assert!(v <= vec[2*i + 2]);
        }
    }
}

fn sift_down<T: PartialOrd>(vec: &mut [T], mut pos: usize) {
    loop {
        let mut child = 2 * pos + 1;
        if child >= vec.len() {
            return
        }
        let right = child + 1;
        // Find smaller child
        if right < vec.len() && vec[right] < vec[child] {
            child = right;
        }
        // If we're in order, do nothing.
        if vec[pos] < vec[child] {
            return
        }
        // Otherwise, swap down and continue
        vec.swap(pos, child);
        pos = child;
    }
}
