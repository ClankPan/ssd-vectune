#[cfg(test)]
mod tests;

pub mod graph;
pub mod point;
pub mod original_vector_reader;

use anyhow::Result;
use bit_set::BitSet;
use point::Point;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::vec;
use vectune::PointInterface;
use original_vector_reader::OriginalVectorReader;

type VectorIndex = usize;
type ClusterPoint = Point;
type PointSum = Point;
type NumInCluster = usize;


fn main() -> Result<()> {

    let seed = 01234;

    /* k-meansをSSD上で行う */
    // 1. memmap
    // 2. 1つの要素ずつアクセスする関数を定義
    let path = "deep1M.fbin";
    let vector_reader = OriginalVectorReader::new(path)?;

    // 3. k個のindexをランダムで決めて、Vec<(ClusterPoint, PointSum, NumInCluster)>
    let mut rng = SmallRng::seed_from_u64(seed);
    let num_clusters: i32 = 16;
    let mut cluster_points: Vec<ClusterPoint> = (0..num_clusters)
        .into_iter()
        .map(|_| {
            let random_index = rng.gen_range(0..vector_reader.get_num_vectors());
            let random_selected_vector = vector_reader.read(&random_index).unwrap();
            Point::from_f32_vec(random_selected_vector)
        })
        .collect();
    // 4. 全ての点に対して、first, secondのclusterを決めて、firstのPointSumに加算。Vec<(FirstLabal, SecondLabel)>
    let mut cluster_labels = vec![(0, 0); vector_reader.get_num_vectors()];
    let dist_threshold = 0.5;
    let max_iter_count = 100;

    for _ in 0..max_iter_count {
        let new_cluster_points: Vec<ClusterPoint> = cluster_labels
            .par_iter_mut()
            .enumerate()
            .map(|(index, (first, second))| {
                let vector = vector_reader.read(&index).unwrap();
                let target_point = Point::from_f32_vec(vector);
                let mut dists: Vec<(u8, f32)> = cluster_points
                    .iter()
                    .enumerate()
                    .map(|(cluster_label, cluster_point)| {
                        (cluster_label as u8, cluster_point.distance(&target_point))
                    })
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Less));
                *first = dists[0].0;
                *second = dists[1].0;
                let mut cluster_sums: Vec<Option<(PointSum, NumInCluster)>> =
                    vec![None; num_clusters as usize];
                cluster_sums[first.clone() as usize] = Some((target_point, 1));
                cluster_sums
            })
            // Only a cluster that have been updated in each iterator are added.
            .reduce_with(|acc, vec| {
                acc.into_iter()
                    .zip(vec.into_iter())
                    .map(|(a, b)| add_each_points(a, b))
                    .collect()
            })
            .unwrap()
            .into_iter()
            .map(|cluster_sums| {
                let (p, n) = cluster_sums.expect("Not a single element belonging to the cluster.");
                let ave: Vec<f32> = p.to_f32_vec().into_iter().map(|x| x / n as f32).collect();
                Point::from_f32_vec(ave)
            })
            .collect();

        // let old_cluster_points = cluster_points.clone();　cluster_points = new_cluster_points;
        let old_cluster_points = std::mem::replace(&mut cluster_points, new_cluster_points);

        // 5. PointSumをNumInClusterで割って、次のClusterPointを求める。　それらの差が閾値以下になった時に終了する。
        let cluster_dists: Vec<f32> = old_cluster_points
            .into_iter()
            .zip(&cluster_points)
            .map(|(a, b)| a.distance(b))
            .collect();
        if cluster_dists.iter().all(|&x| x <= dist_threshold) {
            break;
        }
    }

    fn add_each_points(
        a: Option<(PointSum, NumInCluster)>,
        b: Option<(Point, NumInCluster)>,
    ) -> Option<(PointSum, NumInCluster)> {
        match (a, b) {
            (Some((x_points, x_num)), Some((y_points, y_num))) => {
                Some((x_points.add(&y_points), x_num + y_num))
            }
            (Some(x), None) => Some(x),
            (None, Some(y)) => Some(y),
            (None, None) => None,
        }
    }

    /* sharding */
    // let mut node_written_ssd_bitmap = BitVec::from_elem(vector_reader.num_vectors, false);
    let mut node_written_ssd_bitmap = BitSet::new();

    for cluster_label in 0..(num_clusters as u8) {
        // 1. a cluster labelを持つpointをlist upして、vectune::indexに渡す。
        let table_for_shard_id_to_node_id: Vec<VectorIndex> = cluster_labels
            .iter()
            .enumerate()
            .filter(|(_, (first, second))| *first == cluster_label || *second == cluster_label)
            .map(|(id, _)| id)
            .collect();
        let shard: Vec<Point> = table_for_shard_id_to_node_id
            .par_iter()
            .map(|node_id| Point::from_f32_vec(vector_reader.read(node_id).unwrap()))
            .collect();
        // 2. vectune::indexに渡すノードのindexとidとのtableを作る
        let (indexed_shard, start_shard_id): (Vec<(Point, Vec<u32>)>, u32) =
            vectune::Builder::default().set_seed(seed).build(shard);
        let _start_node_id = table_for_shard_id_to_node_id[start_shard_id as usize];

        // 3. idをもとにssdに書き込む。
        // 4. bitmapを持っておいて、idがtrueの時には、すでにあるedgesをdeserializeして、extend, dup。
        indexed_shard.into_par_iter().enumerate().for_each(
            |(shard_id, (_point, shard_id_edges))| {
                let node_id = table_for_shard_id_to_node_id[shard_id as usize];
                let _edges: Vec<VectorIndex> = shard_id_edges
                    .into_iter()
                    .map(|edge_shard_id| table_for_shard_id_to_node_id[edge_shard_id as usize])
                    .collect();

                if node_written_ssd_bitmap.contains(node_id) {
                    // Pointは書き込まず、元のedgesだけ復号して、追加する
                } else {
                    // Pointとedgesをserializeしてssdに書き込む
                }
            },
        );
        table_for_shard_id_to_node_id.iter().for_each(|node_id| {
            node_written_ssd_bitmap.insert(*node_id);
        });
    }

    /* gordering */
    // 1. node-idでssdから取り出すメソッドを定義する
    // 2. 並び替えの順番をもとに、ssdに書き込む。
    // wip : gordering用のshuffleを消して、別のロジックに書き換える。

    Ok(())
}
