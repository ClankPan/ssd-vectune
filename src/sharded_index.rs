use crate::graph_store::GraphStore;
use crate::k_means::ClusterLabel;
use crate::original_vector_reader::OriginalVectorReaderTrait;
use crate::point::Point;
use crate::storage::Storage;
use crate::VectorIndex;
use bit_vec::BitVec;
use indicatif::ProgressBar;
use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

pub fn sharded_index<R: OriginalVectorReaderTrait + std::marker::Sync>(
    vector_reader: &R,
    graph_on_storage: &GraphStore<Storage>,
    num_clusters: &ClusterLabel,
    cluster_labels: &[(ClusterLabel, ClusterLabel)],
    num_node_in_sector: &usize,
    seed: u64,
) {
    let mut check_node_written: Vec<u8> = vec![0; cluster_labels.len()];
    let mut rng = SmallRng::seed_from_u64(seed);

    // wip
    //　ノードが属するgroupをtupleで記録する。
    // そのために、globalにextendして、その際にoriginalのindexに戻しておく。
    let mut groups: Vec<Vec<u32>> = Vec::new();

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
        let (indexed_shard, start_shard_id, backlinks): (
            Vec<(Point, Vec<u32>)>,
            u32,
            Vec<Vec<u32>>,
        ) = vectune::Builder::default()
            .set_seed(seed)
            // .set_r(20) // wip
            .progress(ProgressBar::new(1000))
            .build(shard_points);
        let _start_node_id = table_for_shard_id_to_node_id[start_shard_id as usize];

        /* Gordering */
        let get_backlinks = |id: &u32| -> Vec<u32> { backlinks[*id as usize].clone() };
        let get_edges = |id: &u32| -> Vec<u32> { indexed_shard[*id as usize].1.clone() }; // originalのnode_indexに変換する前に、そのシャードのみでgorderを行う。
        let target_node_bit_vec = BitVec::from_elem(indexed_shard.len(), true);
        let window_size = *num_node_in_sector;
        let reordered_node_ids: Vec<Vec<u32>> = vectune::gorder(
            get_edges,
            get_backlinks,
            target_node_bit_vec,
            window_size,
            &mut rng,
        )
        .into_iter()
        .map(|group| {
            group
                .into_iter()
                .map(|shard_index| table_for_shard_id_to_node_id[shard_index as usize] as u32)
                .collect()
        })
        .collect();
        groups.extend(reordered_node_ids);

        // 3. idをもとにssdに書き込む。
        // 4. bitmapを持っておいて、idがtrueの時には、すでにあるedgesをdeserializeして、extend, dup。
        /* Merging */

        indexed_shard.into_par_iter().enumerate().for_each(
            |(shard_id, (point, shard_id_edges))| {
                let node_id = table_for_shard_id_to_node_id[shard_id];
                let mut edges: Vec<u32> = shard_id_edges
                    .into_iter()
                    .map(|edge_shard_id| {
                        table_for_shard_id_to_node_id[edge_shard_id as usize] as u32
                    })
                    .collect();

                let original_len = edges.len();

                if check_node_written[node_id] >= 1 {
                    // 元のedgesだけ復号して、追加する
                    edges.extend(graph_on_storage.read_edges(&node_id).unwrap());
                    edges.sort();
                    edges.dedup();

                    // println!("already exsits, node: {}", node_id);
                }

                if edges.len() > 140 {
                    println!(
                        "edge is over {}. node_id: {}, len: {}, original_len: {}",
                        140,
                        node_id,
                        edges.len(),
                        original_len
                    );
                }

                graph_on_storage
                    .write_node(&node_id, &point.to_f32_vec(), &edges)
                    .unwrap();
            },
        );

        table_for_shard_id_to_node_id
            .iter()
            .for_each(|node_id| check_node_written[*node_id] += 1);

        println!(
            "check_node_written is :{}",
            check_node_written.iter().all(|count| *count < 3)
        );
    }

    let belong_groups: Vec<(u32, u32)> = groups
        .iter()
        .enumerate()
        .map(|(group_index, group)| {
            group
                .iter()
                .map(|node_index| (*node_index, group_index as u32))
                .collect::<Vec<(u32, u32)>>()
        })
        .flatten()
        .sorted()
        .chunks(2)
        .into_iter()
        .map(|a| {
            let a: Vec<(u32, u32)> = a.collect();
            (a[0].1, a[1].1)
        })
        .collect();

    


}

fn pickup_target_nodes(
    cluster_label: &ClusterLabel,
    cluster_labels: &[(ClusterLabel, ClusterLabel)],
) -> Vec<VectorIndex> {
    cluster_labels
        .iter()
        .enumerate()
        .filter(|(_, (first, second))| first == cluster_label || second == cluster_label)
        .map(|(node_index, _)| node_index)
        .collect()
}

// WIP:: test sharded_index
#[cfg(test)]
mod test {
    // #[test]
    // fn testing_sharded_index() {}
}
