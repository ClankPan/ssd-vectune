
use crate::point::Point;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::vec;
use vectune::PointInterface;
use crate::original_vector_reader::OriginalVectorReader;

type ClusterPoint = Point;
type PointSum = Point;
type NumInCluster = usize;
type ClusterLabel = u8;

pub fn on_disk_k_means(vector_reader: &OriginalVectorReader, num_clusters: &ClusterLabel, seed: &u64) -> Vec<(ClusterLabel, ClusterLabel)> {

    let mut rng = SmallRng::seed_from_u64(*seed);
    let mut cluster_points: Vec<ClusterPoint> = (0..*num_clusters)
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
                    vec![None; *num_clusters as usize];
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

    cluster_labels
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