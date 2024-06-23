use std::sync::atomic;
use std::sync::atomic::AtomicUsize;

use crate::graph_store::GraphStore;
use crate::k_means::ClusterLabel;
use crate::merge_gorder::merge_gorder;
use crate::original_vector_reader::OriginalVectorReaderTrait;
use crate::point::Point;
use crate::storage::Storage;
use crate::VectorIndex;
use bit_vec::BitVec;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

fn store_and_load<R: OriginalVectorReaderTrait<f32> + std::marker::Sync>(
    prev_result: Option<(Vec<(Point, Vec<u32>)>, Vec<VectorIndex>)>,
    check_node_written: &mut BitVec,
    next_cluster_label: u8,
    cluster_labels: &[(ClusterLabel, ClusterLabel)],
    vector_reader: &R,
    graph_on_storage: &GraphStore<Storage>,
) -> Option<(Vec<Point>, Vec<VectorIndex>)> {
    let pb = ProgressBar::new(1000);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green}  {msg}\n[{elapsed_precise}] {percent:>3}% {wide_bar:.cyan/blue}",
            )
            .unwrap(),
    );

    /* Store previous result indexed points */
    if let Some((prev_indexed_shard, prev_table_for_shard_id_to_node_id)) = prev_result {
        pb.set_length(prev_indexed_shard.len() as u64);
        pb.set_message("storing indexed-points to disk");
        let progress_done = AtomicUsize::new(0);

        prev_indexed_shard.into_par_iter().enumerate().for_each(
            |(shard_id, (point, shard_id_edges))| {
                let node_id = prev_table_for_shard_id_to_node_id[shard_id];
                let mut edges: Vec<u32> = shard_id_edges
                    .into_par_iter()
                    .map(|edge_shard_id| {
                        prev_table_for_shard_id_to_node_id[edge_shard_id as usize] as u32
                    })
                    .collect();

                let original_len = edges.len();

                if check_node_written.get(node_id).unwrap() {
                    // 元のedgesだけ復号して、追加する
                    edges.extend(graph_on_storage.read_edges(&node_id).unwrap());
                    edges.sort();
                    edges.dedup();
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

                let value = progress_done.fetch_add(1, atomic::Ordering::Relaxed);
                if value % 1000 == 0 {
                    pb.set_position(value as u64);
                }
            },
        );

        prev_table_for_shard_id_to_node_id
            .iter()
            .for_each(|node_id| check_node_written.set(*node_id, true));
    }

    /* Load next cluster node points */
    let indexed_shard = if let Some(table_for_shard_id_to_node_id) =
        pickup_target_nodes(&next_cluster_label, cluster_labels)
    {
        pb.set_length(table_for_shard_id_to_node_id.len() as u64);
        pb.set_message("reading points from disk");
        let progress_done = AtomicUsize::new(0);

        let shard_points: Vec<Point> = table_for_shard_id_to_node_id
            .par_iter()
            .map(|node_id| {
                let point = Point::from_f32_vec(vector_reader.read(node_id).unwrap());
                let value = progress_done.fetch_add(1, atomic::Ordering::Relaxed);
                if value % 1000 == 0 {
                    pb.set_position(value as u64);
                }
                point
            })
            .collect();

        Some((shard_points, table_for_shard_id_to_node_id))
    } else {
        None
    };
    pb.finish();

    indexed_shard
}

fn execute_index(
    shard: Option<(Vec<Point>, Vec<VectorIndex>)>,
    seed: u64,
    merge_gorder_groups: &mut Vec<Vec<u32>>,
    num_node_in_sector: &usize,
) -> Option<(Vec<(Point, Vec<u32>)>, Vec<VectorIndex>)> {
    if let Some((shard_points, table_for_shard_id_to_node_id)) = shard {
        let pb = ProgressBar::new(1000);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green}  {msg}\n[{elapsed_precise}] {percent:>3}% {wide_bar:.cyan/blue}").unwrap());

        let (indexed_shard, start_shard_id, backlinks): (
            Vec<(Point, Vec<u32>)>,
            u32,
            Vec<Vec<u32>>,
        ) = vectune::Builder::default()
            .set_seed(seed)
            // .set_r(20) // wip
            .progress(pb)
            .build(shard_points);
        let _start_node_id = table_for_shard_id_to_node_id[start_shard_id as usize];

        let mut rng = SmallRng::seed_from_u64(seed);

        /* Gordering */
        {
            let pb = ProgressBar::new(1000);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green}  {msg}\n[{elapsed_precise}] {percent:>3}% {wide_bar:.cyan/blue}").unwrap());

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
                Some(pb),
            )
            .into_par_iter()
            .map(|group| {
                if group.len() != *num_node_in_sector {
                    println!("{}", group.len())
                }

                group
                    .into_par_iter()
                    .map(|shard_index| table_for_shard_id_to_node_id[shard_index as usize] as u32)
                    .collect()
            })
            .collect();

            // assert_eq!(
            //     reordered_node_ids.iter().flatten().count(),
            //     indexed_shard.len()
            // );

            merge_gorder_groups.extend(reordered_node_ids);
        }

        Some((indexed_shard, table_for_shard_id_to_node_id))
    } else {
        None
    }
}

pub fn sharded_index<R: OriginalVectorReaderTrait<f32> + std::marker::Sync>(
    vector_reader: &R,
    graph_on_storage: &GraphStore<Storage>,
    num_clusters: &ClusterLabel,
    cluster_labels: &[(ClusterLabel, ClusterLabel)],
    num_node_in_sector: &usize,
    seed: u64,
) -> Vec<Vec<u32>> {
    // let mut check_node_written: Vec<u8> = vec![0; cluster_labels.len()];
    let mut check_node_written = BitVec::from_elem(cluster_labels.len(), false);
    let mut rng = SmallRng::seed_from_u64(seed);

    println!("num_node_in_sector: {}", num_node_in_sector);

    // wip
    //　ノードが属するgroupをtupleで記録する。
    // そのために、globalにextendして、その際にoriginalのindexに戻しておく。
    let mut merge_gorder_groups: Vec<Vec<u32>> = Vec::new();

    let mut shard_for_execution = None;
    let mut indexed_shard_for_storing = None;

    for cluster_label in 0..*num_clusters + 2 {
        println!("shard: {}", cluster_label);

        // let next_shard = store_and_load(
        //     indexed_shard_for_storing,
        //     &mut check_node_written,
        //     cluster_label,
        //     cluster_labels,
        //     vector_reader,
        //     graph_on_storage,
        // );

        // let executed_indexed_shard =
        //     execute_index(shard_for_execution, seed, &mut merge_gorder_groups, num_node_in_sector);

        let (next_shard, executed_indexed_shard) = rayon::join(
            || {
                store_and_load(
                    indexed_shard_for_storing,
                    &mut check_node_written,
                    cluster_label,
                    cluster_labels,
                    vector_reader,
                    graph_on_storage,
                )
            },
            || {
                execute_index(
                    shard_for_execution,
                    seed,
                    &mut merge_gorder_groups,
                    num_node_in_sector,
                )
            },
        );

        shard_for_execution = next_shard;
        indexed_shard_for_storing = executed_indexed_shard;

        println!("\n\n");
    }

    let belong_groups: Vec<(u32, u32)> = merge_gorder_groups
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

    let get_edges = |id: &u32| -> Vec<u32> {
        let (group_a, group_b) = belong_groups[*id as usize];
        let mut member: Vec<u32> = Vec::new();
        member.extend(merge_gorder_groups[group_a as usize].clone());
        member.extend(merge_gorder_groups[group_b as usize].clone());

        member
    };

    let target_node_bit_vec = BitVec::from_elem(belong_groups.len(), true);
    let window_size = *num_node_in_sector;
    println!("do vectune::gorder");
    let reordered_node_ids = merge_gorder(get_edges, target_node_bit_vec, window_size, &mut rng);

    reordered_node_ids
}

fn pickup_target_nodes(
    cluster_label: &ClusterLabel,
    cluster_labels: &[(ClusterLabel, ClusterLabel)],
) -> Option<Vec<VectorIndex>> {
    let table: Vec<VectorIndex> = cluster_labels
        .iter()
        .enumerate()
        .filter(|(_, (first, second))| first == cluster_label || second == cluster_label)
        .map(|(node_index, _)| node_index)
        .collect();
    if table.len() == 0 {
        None
    } else {
        Some(table)
    }
}

// WIP:: test sharded_index
#[cfg(test)]
mod test {
    // #[test]
    // fn testing_sharded_index() {}
}
