use crate::original_vector_reader::OriginalVectorReaderTrait;
use crate::point::Point;
use bytesize::GB;
use indicatif::ProgressBar;
use rand::rngs::SmallRng;
use rand::thread_rng;
use rand::Rng;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::sync::atomic;
use std::sync::atomic::AtomicUsize;
use std::vec;
use vectune::PointInterface;

pub type VectorIndex = usize;
pub type ClusterPoint = Point;
pub type PointSum = Point;
pub type NumInCluster = usize;
pub type ClusterLabel = u8;
pub type DistAndNode = (f32, VectorIndex);
pub type ClosedVectorIndex = VectorIndex;

// struct NodeLable {
//     index: VectorIndex,
//     first: ClusterLabel,
//     second: ClusterLabel
// }

pub fn on_disk_k_means<R: OriginalVectorReaderTrait<f32> + std::marker::Sync>(
    vector_reader: &mut R,
    num_clusters: &ClusterLabel,
    max_chunk_giga_byte_size: u64,
    rng: &mut SmallRng,
) -> (
    Vec<(ClusterLabel, ClusterLabel)>,
    Vec<(ClusterPoint, ClosedVectorIndex)>,
) {
    assert!(*num_clusters > 2);

    println!(
        "max_chunk_giga_byte_size: {} GiB ({} Byte)",
        max_chunk_giga_byte_size,
        max_chunk_giga_byte_size * GB
    );
    let max_chunk_byte_size = (max_chunk_giga_byte_size * GB) as usize;

    let mut cluster_points: Vec<(ClusterPoint, ClosedVectorIndex)> = (0..*num_clusters)
        .map(|_| {
            let random_index = rng.gen_range(0..vector_reader.get_num_vectors());
            let random_selected_vector = vector_reader.read(&random_index).unwrap();
            (Point::from_f32_vec(random_selected_vector), random_index)
        })
        .collect();
    // 4. 全ての点に対して、first, secondのclusterを決めて、firstのPointSumに加算。Vec<(FirstLabal, SecondLabel)>
    let mut cluster_labels = vec![(0, 0); vector_reader.get_num_vectors()];
    let dist_threshold = 0.01;
    let max_iter_count = 30;
    let mut iter_count = 0;

    // let max_chunk_byte_size = 20 * GB as usize;
    let one_vector_byte_size = vector_reader.get_vector_dim() * core::mem::size_of::<f32>();
    println!("one_vector_byte_size: {} Byte", one_vector_byte_size);
    let num_vectos_in_chunk = max_chunk_byte_size / one_vector_byte_size;
    // let num_vectos_in_chunk = max_chunk_byte_size / (vector_reader.get_vector_dim() * 4);
    println!("num_vectos_in_chunk: {}", num_vectos_in_chunk);

    for _ in 0..max_iter_count {
        println!("k-means iter count: {}", iter_count);

        let progress = Some(ProgressBar::new(1000));
        let progress_done = AtomicUsize::new(0);
        if let Some(bar) = &progress {
            bar.set_length(cluster_labels.len() as u64);
            bar.set_message("labelling rest vectors");
        }

        iter_count += 1;
        let new_cluster_points = (0..vector_reader.get_num_vectors())
            .step_by(num_vectos_in_chunk)
            .map(|start| {
                let end =
                    std::cmp::min(start + num_vectos_in_chunk, vector_reader.get_num_vectors());
                let chunk_vectors = vector_reader.read_with_range(&start, &end).unwrap();
                assert_eq!(chunk_vectors.len(), end - start);
                let slice = &mut cluster_labels[start..end];
                let cluster_sums_in_chunk: Vec<Option<(PointSum, NumInCluster, DistAndNode)>> =
                    slice
                        .par_iter_mut()
                        .enumerate()
                        .map(|(chunk_index, (first_label, second_label))| {
                            let original_vector_index = chunk_index + start;
                            let vector = chunk_vectors[chunk_index].clone();
                            let target_point = Point::from_f32_vec(vector);
                            let dists: Vec<(u8, f32)> = cluster_points
                                .par_iter()
                                .enumerate()
                                .map(|(cluster_label, (cluster_point, _))| {
                                    (cluster_label as u8, cluster_point.distance(&target_point))
                                })
                                .collect();

                            let (first_label_and_dist, second_label_and_dist) =
                                find_two_smallest(&dists);
                            *first_label = first_label_and_dist.0;
                            *second_label = second_label_and_dist.0;

                            assert_ne!(first_label, second_label);

                            let _num_clusters = *num_clusters as usize;
                            let mut cluster_sums: Vec<
                                Option<(PointSum, NumInCluster, DistAndNode)>,
                            > = vec![None; _num_clusters];
                            cluster_sums[*first_label as usize] = Some((
                                target_point,
                                1,
                                (first_label_and_dist.1, original_vector_index),
                            ));

                            if let Some(bar) = &progress {
                                let value = progress_done.fetch_add(1, atomic::Ordering::Relaxed);
                                if value % 1000 == 0 {
                                    bar.set_position(value as u64);
                                }
                            }

                            cluster_sums
                        })
                        // Only a cluster that have been updated in each iterator are added.
                        .reduce_with(add_cluster_sums)
                        .unwrap();
                cluster_sums_in_chunk
            })
            // .reduce_with(add_cluster_sums)
            .reduce(add_cluster_sums)
            .unwrap()
            .into_par_iter() // ToDo: parallel
            .map(|cluster_sums| {
                let mut rng = thread_rng(); // wip no seedable
                match cluster_sums {
                    Some(cluster_sums) => {
                        let (p, n, (_, closed_index)) = cluster_sums;
                        let ave: Vec<f32> = p
                            .to_f32_vec()
                            .into_par_iter()
                            .map(|x| x / n as f32)
                            .collect();
                        // closed_indexは、実際にはPoint::from_f32_vec(ave)に近いvectorのindexとなるわけではないが、
                        // 終了条件としてクラスターの差が閾値以下になることを前提としているので、closed_indexを利用する。
                        (Point::from_f32_vec(ave), closed_index)
                    }
                    None => {
                        // When if If none of vectors belonged to a cluster,
                        let random_index = rng.gen_range(0..vector_reader.get_num_vectors());
                        let random_selected_vector = vector_reader.read(&random_index).unwrap();
                        (Point::from_f32_vec(random_selected_vector), random_index)
                    }
                }
            })
            .collect();

        // let old_cluster_points = cluster_points.clone();　cluster_points = new_cluster_points;
        let old_cluster_points = std::mem::replace(&mut cluster_points, new_cluster_points);

        // 5. PointSumをNumInClusterで割って、次のClusterPointを求める。　それらの差が閾値以下になった時に終了する。
        let cluster_dists: Vec<f32> = old_cluster_points
            .into_par_iter()
            .zip(&cluster_points)
            .map(|((a, _), (b, _))| a.distance(b))
            .collect();

        if cluster_dists.iter().all(|&x| x <= dist_threshold) {
            break;
        }
    }
    (cluster_labels, cluster_points)
}

fn add_cluster_sums(
    acc: Vec<Option<(PointSum, NumInCluster, DistAndNode)>>,
    vec: Vec<Option<(PointSum, NumInCluster, DistAndNode)>>,
) -> Vec<Option<(PointSum, NumInCluster, DistAndNode)>> {
    acc.into_par_iter()
        .zip(vec.into_par_iter())
        .map(|(a, b)| add_each_points(a, b))
        .collect()
}

fn add_each_points(
    a: Option<(PointSum, NumInCluster, DistAndNode)>,
    b: Option<(Point, NumInCluster, DistAndNode)>,
) -> Option<(PointSum, NumInCluster, DistAndNode)> {
    match (a, b) {
        (Some((x_points, x_num, x_dist_and_index)), Some((y_points, y_num, y_dist_and_index))) => {
            Some((
                x_points.add(&y_points),
                x_num + y_num,
                min_by_dist(x_dist_and_index, y_dist_and_index),
            ))
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

fn min_by_dist(a: DistAndNode, b: DistAndNode) -> DistAndNode {
    if a.0 < b.0 {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use crate::k_means::on_disk_k_means;
    use crate::original_vector_reader::OriginalVectorReader;
    use bytesize::GB;
    use rand::{rngs::SmallRng, SeedableRng};

    #[test]
    fn testing_on_disk_k_means() {
        let mut vector_reader =
            OriginalVectorReader::new_with("test_vectors/base.10M.fbin", 1).unwrap();
        let num_clusters: u8 = 16;
        let mut rng = SmallRng::seed_from_u64(rand::random());
        let (cluster_labels, cluster_points) =
            on_disk_k_means(&mut vector_reader, &num_clusters, 1 * GB, &mut rng);

        assert!(cluster_labels
            .iter()
            .all(|(a, b)| (a < &num_clusters) && (b < &num_clusters)));
        let deplicateds = cluster_labels
            .iter()
            .enumerate()
            .filter(|(_i, (a, b))| a == b)
            .map(|(i, (a, b))| (i, *a, *b))
            .collect::<Vec<(usize, u8, u8)>>();
        println!("{:?}", deplicateds);
        assert_eq!(deplicateds.len(), 0);
        // let collect_nodes = BitVec::from_elem(vector_reader.get_num_vectors(), false);

        // wip assertion

        println!("{:?}", cluster_labels);
    }
}
