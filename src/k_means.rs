use crate::original_vector_reader::OriginalVectorReaderTrait;
use crate::point::Point;
use rand::rngs::SmallRng;
use rand::Rng;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::vec;
use vectune::PointInterface;

pub type ClusterPoint = Point;
pub type PointSum = Point;
pub type NumInCluster = usize;
pub type ClusterLabel = u8;

pub fn on_disk_k_means<R: OriginalVectorReaderTrait + std::marker::Sync>(
    vector_reader: &R,
    num_clusters: &ClusterLabel,
    rng: &mut SmallRng,
) -> Vec<(ClusterLabel, ClusterLabel)> {
    assert!(*num_clusters > 2);

    let mut cluster_points: Vec<ClusterPoint> = (0..*num_clusters)
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
                let dists: Vec<(u8, f32)> = cluster_points
                    .iter()
                    .enumerate()
                    .map(|(cluster_label, cluster_point)| {
                        (cluster_label as u8, cluster_point.distance(&target_point))
                    })
                    .collect();
                // WIP: Test this assignment of mutable to *first and *second
                let (_first, _second) = find_two_smallest(&dists);
                *first = _first.0;
                *second = _second.0;
                let _num_clusters = *num_clusters as usize;
                let mut cluster_sums: Vec<Option<(PointSum, NumInCluster)>> =
                    vec![None; _num_clusters];
                cluster_sums[*first as usize] = Some((target_point, 1));
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

fn find_two_smallest(dists: &[(u8, f32)]) -> ((u8, f32), (u8, f32)) {
    assert!(
        dists.len() >= 2,
        "The dists array must contain at least two elements."
    );

    // Initialize smallest and second_smallest with the first two elements
    let (smallest, second_smallest) = if dists[0].1 < dists[1].1 {
        (dists[0], dists[1])
    } else {
        (dists[1], dists[0])
    };

    // Fold over the rest of the elements
    let result = dists.iter().skip(2).fold(
        (smallest, second_smallest),
        |(smallest, second_smallest), &current| {
            if current.1 < smallest.1 {
                (current, smallest)
            } else if current.1 < second_smallest.1 {
                (smallest, current)
            } else {
                (smallest, second_smallest)
            }
        },
    );

    result
}

#[cfg(test)]
mod tests {
    use crate::k_means::on_disk_k_means;
    use crate::{original_vector_reader::OriginalVectorReaderTrait, VectorIndex};
    use anyhow::Result;
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    // const SEED: u64 = rand::random();

    struct TestVectorReader {
        num_vectors: usize,
        vector_dim: usize,
        vectors: Vec<Vec<f32>>,
    }
    impl TestVectorReader {
        fn new() -> Self {
            let num_vectors = 1000;
            let vector_dim = 96;
            let mut rng = SmallRng::seed_from_u64(rand::random());
            Self {
                num_vectors,
                vector_dim,
                vectors: (0..num_vectors)
                    .map(|_| (0..vector_dim).map(|_| rng.gen::<f32>()).collect())
                    .collect(),
            }
        }
    }
    impl OriginalVectorReaderTrait for TestVectorReader {
        fn read(&self, index: &VectorIndex) -> Result<Vec<f32>> {
            let vector = &self.vectors[*index];
            Ok(vector.clone())
        }

        fn get_num_vectors(&self) -> usize {
            self.num_vectors
        }

        fn get_vector_dim(&self) -> usize {
            self.vector_dim
        }
    }

    #[test]
    fn testing_on_disk_k_means() {
        let vector_reader = TestVectorReader::new();
        let num_clusters: u8 = 16;
        let mut rng = SmallRng::seed_from_u64(rand::random());
        let cluster_labels = on_disk_k_means(&vector_reader, &num_clusters, &mut rng);

        // wip assertion

        println!("{:?}", cluster_labels);
    }
}
