#[cfg(test)]
mod tests;

pub mod graph;
pub mod point;

use anyhow::anyhow;
use anyhow::Result;
use memmap2::Mmap;
use point::Point;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::fs::File;
use std::vec;
use vectune::PointInterface;

type VectorIndex = usize;
type ClusterPoint = Point;
type PointSum = Point;
type NumInCluster = usize;

fn main() -> Result<()> {
    println!("Hello, world!");

    let seed = 01234;

    /* k-meansをSSD上で行う */
    // 1. memmap
    // 2. 1つの要素ずつアクセスする関数を定義
    struct StorageVectorReader {
        mmap: Mmap,
        num_vectors: usize,
        vector_dim: usize,
        start_offset: usize,
    }
    impl StorageVectorReader {
        fn new(path: &str) -> Result<Self> {
            let file = File::open(path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            let num_vectors = u32::from_le_bytes(mmap[0..4].try_into()?) as usize;
            let vector_dim = u32::from_le_bytes(mmap[4..8].try_into()?) as usize;
            let start_offset = 8;

            Ok(Self {
                mmap,
                num_vectors,
                vector_dim,
                start_offset,
            })
        }

        fn read(&self, index: &VectorIndex) -> Result<Vec<f32>> {
            let start = self.start_offset + index * self.vector_dim * 4;
            let end = start + self.vector_dim * 4;
            let bytes = &self.mmap[start..end];
            let vector: Vec<f32> = bytemuck::try_cast_slice(bytes)
                .map_err(|e| anyhow!("PodCastError: {:?}", e))?
                .to_vec();
            Ok(vector)
        }
    }
    let path = "deep1M.fbin";
    let vector_reader = StorageVectorReader::new(path)?;

    // 3. k個のindexをランダムで決めて、Vec<(ClusterPoint, PointSum, NumInCluster)>
    let mut rng = SmallRng::seed_from_u64(seed);
    let num_clusters: i32 = 16;
    let cluster_points: Vec<ClusterPoint> = (0..num_clusters)
        .into_iter()
        .map(|_| {
            let random_index = rng.gen_range(0..vector_reader.num_vectors);
            let random_selected_vector = vector_reader.read(&random_index).unwrap();
            Point::from_f32_vec(random_selected_vector)
        })
        .collect();
    // 4. 全ての点に対して、first, secondのclusterを決めて、firstのPointSumに加算。Vec<(FirstLabal, SecondLabel)>
    let mut cluster_labels = vec![(0, 0); vector_reader.num_vectors];
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

    // 5. PointSumをNumInClusterで割って、次のClusterPointを求める。　それらの差が閾値以下になった時に終了する。
    let cluster_dists: Vec<f32> = cluster_points.into_iter().zip(&new_cluster_points).map(|(a, b)| a.distance(b)).collect();

    /* sharding */
    // 1. a cluster labelを持つpointをlist upして、vectune::indexに渡す。
    // 2. vectune::indexに渡すノードのindexとidとのtableを作る
    // 3. idをもとにssdに書き込む。
    // 4. bitmapを持っておいて、idがtrueの時には、すでにあるedgesをdeserializeして、extend, dup。

    /* gordering */
    // 1. node-idでssdから取り出すメソッドを定義する
    // 2. 並び替えの順番をもとに、ssdに書き込む。

    Ok(())
}
