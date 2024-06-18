use crate::original_vector_reader::OriginalVectorReaderTrait;
use crate::point::Point;
use bytesize::GB;
use indicatif::ProgressBar;
use rand::rngs::SmallRng;
use rand::Rng;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::sync::atomic;
use std::sync::atomic::AtomicUsize;
use std::vec;
use rand::seq::SliceRandom;
use vectune::PointInterface;

pub type VectorIndex = usize;
pub type ClusterPoint = Point;
pub type PointSum = Point;
pub type NumInCluster = usize;
pub type ClusterLabel = u8;
pub type DistAndNode = (f32, VectorIndex);
pub type ClosedVectorIndex = VectorIndex;


struct NodeLable {
    index: VectorIndex,
    first: ClusterLabel,
    second: ClusterLabel
}

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
    let num_vectos_in_chunk = max_chunk_byte_size / (vector_reader.get_vector_dim() * 4);
    println!("num_vectos_in_chunk: {}", num_vectos_in_chunk);

    /*
    indexをランダムな順番に並び替えたものを用意する。
    そこから先頭の20GiB分をssdから取り出してVec<Point>を作る。
    - Vec<Point>, Vec<(original_index, label1, label2)>
    Vec<Point>に対してクラスタリングを行う。
    ssdに残されているデータポイントに対して、クラスターを決める。
    */

    // let mut vector_labels: Vec<NodeLable> = (0..vector_reader.get_num_vectors()).map(|index| NodeLable {
    //     index,
    //     first: 0,
    //     second: 0,
    // }).collect();
    // vector_labels.shuffle(rng);

    // let sample_labels = &mut vector_labels[0..num_vectos_in_chunk];

    // println!("sampling vectors");
    // let progress = Some(ProgressBar::new(1000));
    // let progress_done = AtomicUsize::new(0);
    // if let Some(bar) = &progress {
    //     bar.set_length(sample_labels.len() as u64);
    //     bar.set_message("sampling vectors");
    // }

    // let sample_vectors: Vec<Vec<f32>> = sample_labels.par_iter().map(|labels| {
    //     let vector = vector_reader.read(&labels.index).unwrap();

    //     if let Some(bar) = &progress {
    //         let value = progress_done.fetch_add(1, atomic::Ordering::Relaxed);
    //         if value % 1000 == 0 {
    //             bar.set_position(value as u64);
    //         }
    //     }

    //     vector
    // }).collect();

    // if let Some(bar) = &progress {
    //     bar.finish();
    // }


    // for _ in 0..max_iter_count {
    //     println!("k-means iter count: {}", iter_count);

    //     let progress = Some(ProgressBar::new(1000));
    //     let progress_done = AtomicUsize::new(0);
    //     if let Some(bar) = &progress {
    //         bar.set_length(sample_labels.len() as u64);
    //         bar.set_message("on disk sampling k-means");
    //     }
    
    //     iter_count += 1;
    //     let new_cluster_points: Vec<(ClusterPoint, ClosedVectorIndex)> = sample_labels
    //         .par_iter_mut()
    //         .enumerate()
    //         .map(|(sample_vector_index, labels)| {
    //             let vector = sample_vectors[sample_vector_index].clone();
    //             let target_point = Point::from_f32_vec(vector);
    //             let dists: Vec<(u8, f32)> = cluster_points
    //                 .iter()
    //                 .enumerate()
    //                 .map(|(cluster_label, (cluster_point, _))| {
    //                     (cluster_label as u8, cluster_point.distance(&target_point))
    //                 })
    //                 .collect();
    //             // WIP: Test this assignment of mutable to *first and *second
    //             let (first_label_and_dist, second_label_and_dist) = find_two_smallest(&dists);
    //             labels.first = first_label_and_dist.0;
    //             labels.second = second_label_and_dist.0;
    //             let _num_clusters = *num_clusters as usize;
    //             let mut cluster_sums: Vec<Option<(PointSum, NumInCluster, DistAndNode)>> =
    //                 vec![None; _num_clusters];
    //             cluster_sums[labels.first as usize] =
    //                 Some((target_point, 1, (first_label_and_dist.1, labels.index)));

    //             if let Some(bar) = &progress {
    //                 let value = progress_done.fetch_add(1, atomic::Ordering::Relaxed);
    //                 if value % 1000 == 0 {
    //                     bar.set_position(value as u64);
    //                 }
    //             }

    //             cluster_sums
    //         })
    //         // Only a cluster that have been updated in each iterator are added.
    //         .reduce_with(|acc, vec| {
    //             acc.into_iter()
    //                 .zip(vec.into_iter())
    //                 .map(|(a, b)| add_each_points(a, b))
    //                 .collect()
    //         })
    //         .unwrap()
    //         .into_iter() // ToDo: parallel
    //         .map(|cluster_sums| {
    //             match cluster_sums {
    //                 Some(cluster_sums) => {
    //                     let (p, n, (_, closed_index)) = cluster_sums;
    //                     let ave: Vec<f32> =
    //                         p.to_f32_vec().into_iter().map(|x| x / n as f32).collect();
    //                     // closed_indexは、実際にはPoint::from_f32_vec(ave)に近いvectorのindexとなるわけではないが、
    //                     // 終了条件としてクラスターの差が閾値以下になることを前提としているので、closed_indexを利用する。
    //                     (Point::from_f32_vec(ave), closed_index)
    //                 }
    //                 None => {
    //                     // When if If none of vectors belonged to a cluster,
    //                     let random_index = rng.gen_range(0..vector_reader.get_num_vectors());
    //                     let random_selected_vector = vector_reader.read(&random_index).unwrap();
    //                     (Point::from_f32_vec(random_selected_vector), random_index)
    //                 }
    //             }
    //         })
    //         .collect();

    //     // let old_cluster_points = cluster_points.clone();　cluster_points = new_cluster_points;
    //     let old_cluster_points = std::mem::replace(&mut cluster_points, new_cluster_points);

    //     // 5. PointSumをNumInClusterで割って、次のClusterPointを求める。　それらの差が閾値以下になった時に終了する。
    //     let cluster_dists: Vec<f32> = old_cluster_points
    //         .into_iter()
    //         .zip(&cluster_points)
    //         .map(|((a, _), (b, _))| a.distance(b))
    //         .collect();
    //     if cluster_dists.iter().all(|&x| x <= dist_threshold) {
    //         break;
    //     }
    // }


    // let rest_labels = &mut vector_labels[num_vectos_in_chunk..];

    // let progress = Some(ProgressBar::new(1000));
    // let progress_done = AtomicUsize::new(0);
    // if let Some(bar) = &progress {
    //     bar.set_length(rest_labels.len() as u64);
    //     bar.set_message("labelling rest vectors");
    // }

    // rest_labels.par_iter_mut().for_each(|labels| {
    //     let vector = vector_reader.read(&labels.index).unwrap();
    //     let target_point = Point::from_f32_vec(vector);
    //     let dists: Vec<(u8, f32)> = cluster_points
    //         .par_iter()
    //         .enumerate()
    //         .map(|(cluster_label, (cluster_point, _))| {
    //             (cluster_label as u8, cluster_point.distance(&target_point))
    //         })
    //         .collect();
    //     // WIP: Test this assignment of mutable to *first and *second
    //     let (first_label_and_dist, second_label_and_dist) = find_two_smallest(&dists);
    //     labels.first = first_label_and_dist.0;
    //     labels.second = second_label_and_dist.0;

    //     if let Some(bar) = &progress {
    //         let value = progress_done.fetch_add(1, atomic::Ordering::Relaxed);
    //         if value % 1000 == 0 {
    //             bar.set_position(value as u64);
    //         }
    //     }
    // });

    // if let Some(bar) = &progress {
    //     bar.finish();
    // }


    // // Sort by orifinal index
    // vector_labels.sort_by(|a, b| a.index.cmp(&b.index));

    // let cluster_labels = vector_labels.into_par_iter().map(|labels| {
    //     (labels.first, labels.second)
    // }).collect();

    // (cluster_labels, cluster_points)


/*----------------------- */


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
            // .collect::<Vec<usize>>()
            // .into_par_iter()
            .map(|start| {
                let end = std::cmp::min(
                    start + num_vectos_in_chunk - 1,
                    vector_reader.get_num_vectors(),
                );
                let chunk_vectors = vector_reader.read_with_range(&start, &end).unwrap();
                let slice = &mut cluster_labels[start..end];
                let cluster_sums_in_chunk: Vec<Option<(PointSum, NumInCluster, DistAndNode)>> = slice
                    .par_iter_mut()
                    .enumerate()
                    .map(|(chunk_index, (first_label, second_label))| {
                        let original_vector_index = chunk_index + start;
                        let vector = chunk_vectors[chunk_index].clone();
                        let target_point = Point::from_f32_vec(vector);
                        let dists: Vec<(u8, f32)> = cluster_points
                            .iter()
                            .enumerate()
                            .map(|(cluster_label, (cluster_point, _))| {
                                (cluster_label as u8, cluster_point.distance(&target_point))
                            })
                            .collect();

                        let (first_label_and_dist, second_label_and_dist) =
                            find_two_smallest(&dists);
                        *first_label = first_label_and_dist.0;
                        *second_label = second_label_and_dist.0;

                        let _num_clusters = *num_clusters as usize;
                        let mut cluster_sums: Vec<Option<(PointSum, NumInCluster, DistAndNode)>> =
                            vec![None; _num_clusters];
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
            .into_iter() // ToDo: parallel
            .map(|cluster_sums| {
                match cluster_sums {
                    Some(cluster_sums) => {
                        let (p, n, (_, closed_index)) = cluster_sums;
                        let ave: Vec<f32> =
                            p.to_f32_vec().into_iter().map(|x| x / n as f32).collect();
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
            .into_iter()
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
    acc.into_iter()
        .zip(vec.into_iter())
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
    use crate::{original_vector_reader::OriginalVectorReaderTrait, VectorIndex};
    use anyhow::Result;
    use bytesize::GB;
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
    impl OriginalVectorReaderTrait<f32> for TestVectorReader {
        fn read(&self, index: &VectorIndex) -> Result<Vec<f32>> {
            let vector = &self.vectors[*index];
            Ok(vector.clone())
        }

        fn read_with_range(&mut self, start: &VectorIndex, end: &VectorIndex) -> Result<Vec<Vec<f32>>> {
            let vector = self.vectors[*start..*end].to_vec();
            Ok(vector)
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
        let mut vector_reader = TestVectorReader::new();
        let num_clusters: u8 = 16;
        let mut rng = SmallRng::seed_from_u64(rand::random());
        let cluster_labels = on_disk_k_means(&mut vector_reader, &num_clusters, 1 * GB, &mut rng);

        // wip assertion

        println!("{:?}", cluster_labels);
    }
}
