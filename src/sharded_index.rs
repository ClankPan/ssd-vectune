use crate::graph_store::GraphStore;
use crate::k_means::ClusterLabel;
use crate::original_vector_reader::OriginalVectorReaderTrait;
use crate::point::Point;
use crate::storage::Storage;
use crate::VectorIndex;
use bit_set::BitSet;
use indicatif::ProgressBar;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

pub fn sharded_index<R: OriginalVectorReaderTrait + std::marker::Sync>(
    vector_reader: &R,
    graph_on_storage: &GraphStore<Storage>,
    num_clusters: &ClusterLabel,
    cluster_labels: &[(ClusterLabel, ClusterLabel)],
    seed: u64,
) {
    let mut node_written_ssd_bitmap = BitSet::new();

    for cluster_label in 0..*num_clusters {
        println!("shard: {}", cluster_label);
        // 1. a cluster labelを持つpointをlist upして、vectune::indexに渡す。
        let table_for_shard_id_to_node_id: Vec<VectorIndex> =
            pickup_target_nodes(&cluster_label, cluster_labels);
        let shard_points: Vec<Point> = table_for_shard_id_to_node_id
            .par_iter()
            .map(|node_id| Point::from_f32_vec(vector_reader.read(node_id).unwrap()))
            .collect();
        // 2. vectune::indexに渡すノードのindexとidとのtableを作る
        let (indexed_shard, start_shard_id): (Vec<(Point, Vec<u32>)>, u32) =
            vectune::Builder::default()
                .set_seed(seed)
                .progress(ProgressBar::new(1000))
                .build(shard_points);
        let _start_node_id = table_for_shard_id_to_node_id[start_shard_id as usize];

        // 3. idをもとにssdに書き込む。
        // 4. bitmapを持っておいて、idがtrueの時には、すでにあるedgesをdeserializeして、extend, dup。
        indexed_shard
            .into_iter()
            .enumerate()
            .for_each(|(shard_id, (point, shard_id_edges))| {
                let node_id = table_for_shard_id_to_node_id[shard_id];
                let mut edges: Vec<u32> = shard_id_edges
                    .into_iter()
                    .map(|edge_shard_id| {
                        table_for_shard_id_to_node_id[edge_shard_id as usize] as u32
                    })
                    .collect();

                if node_written_ssd_bitmap.contains(node_id) {
                    // 元のedgesだけ復号して、追加する
                    edges.extend(graph_on_storage.read_edges(&node_id).unwrap());
                    edges.sort();
                    edges.dedup();
                }

                graph_on_storage
                    .write_node(&node_id, &point.to_f32_vec(), &edges)
                    .unwrap();
            });
        table_for_shard_id_to_node_id.iter().for_each(|node_id| {
            node_written_ssd_bitmap.insert(*node_id);
        });
    }
}

fn pickup_target_nodes(
    cluster_label: &ClusterLabel,
    cluster_labels: &[(ClusterLabel, ClusterLabel)],
) -> Vec<VectorIndex> {
    cluster_labels
        .iter()
        .enumerate()
        .filter(|(_, (first, second))| first == cluster_label || second == cluster_label)
        .map(|(id, _)| id)
        .collect()
}

// WIP:: test sharded_index
#[cfg(test)]
mod test {
    #[test]
    fn testing_sharded_index() {}
}
