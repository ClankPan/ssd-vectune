#![cfg_attr(feature = "simd", feature(portable_simd))]

#[cfg(test)]
mod tests;

pub mod cache;
pub mod graph;
pub mod graph_store;
pub mod k_means;
pub mod merge_gorder;
pub mod original_vector_reader;
pub mod point;
pub mod sharded_index;
pub mod single_index;
pub mod storage;
pub mod utils;

use std::{
    fs::File,
    io::{BufReader, Write},
    path::Path,
    sync::atomic::{self, AtomicUsize},
};

use anyhow::Result;
// use bit_vec::BitVec;
use bytesize::KB;
// use ext_sort::{buffer::LimitedBufferBuilder, ExternalSorter, ExternalSorterBuilder};
use graph::Graph;
use indicatif::ProgressBar;
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use k_means::on_disk_k_means;
// use node_reader::{EdgesIterator, GraphOnStorage, GraphOnStorageTrait};
use original_vector_reader::{OriginalVectorReader, OriginalVectorReaderTrait};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use sharded_index::sharded_index;
// use single_index::single_index;
use vectune::PointInterface;

use crate::{graph_store::GraphStore, point::Point, storage::Storage};

type VectorIndex = usize;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Executes the build process, including merge-index and gorder
    Build {
        /// Skip the merge-index step
        #[arg(long)]
        skip_merge_index: bool,

        #[arg(long)]
        skip_make_backlinks: bool,

        original_vector_path: String,

        max_sector_k_byte_size: usize,
    },
    /// Executes the search process
    Search,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let seed = 01234; // wip
    let mut rng = SmallRng::seed_from_u64(seed);

    // let edge_max_digree_in_merge_indexing = vectune::DEFAULT_R;

    match cli.command {
        Commands::Build {
            skip_merge_index,
            skip_make_backlinks, // wip
            original_vector_path,
            max_sector_k_byte_size,
        } => {
            let max_sector_byte_size = max_sector_k_byte_size * KB as usize;

            /* file path */
            let (
                unordered_graph_storage_path,
                ordered_graph_storage_path,
                graph_json_path,
                cluster_points_path,
                query_vector_path,
                groundtruth_path,
                _backlinks_path,
            ) = if let Some(directory) = Path::new(&original_vector_path).parent() {
                (
                    directory
                        .join("unordered_graph.10M.graph")
                        .display()
                        .to_string(),
                    directory
                        .join("ordered_graph.10M.graph")
                        .display()
                        .to_string(),
                    directory.join("graph.json").display().to_string(),
                    directory.join("cluster_points.json").display().to_string(),
                    directory
                        .join("query.public.10K.fbin")
                        .display()
                        .to_string(),
                    directory
                        .join("groundtruth.public.10K.ibin")
                        .display()
                        .to_string(),
                    directory.join("backlinks.json").display().to_string(),
                )
            } else {
                panic!()
            };

            /* read original vectors */
            println!("reading vector file");
            // 10Mだと大きすぎるので、小さなデータセットをここから作る。
            #[cfg(not(feature = "1m"))]
            let vector_reader = OriginalVectorReader::new(&original_vector_path)?;
            #[cfg(feature = "1m")]
            let vector_reader = OriginalVectorReader::new_with_1m(&original_vector_path)?;

            /* sharding */
            println!("initializing graph");
            let node_byte_size = (vector_reader.get_vector_dim() * 4 + 140 * 4 + 4) as usize;
            let file_byte_size = vector_reader.get_num_vectors() * node_byte_size;
            // let file_byte_size = 11 * 1000000 * node_byte_size;
            let num_node_in_sector = max_sector_byte_size / node_byte_size;
            let sector_byte_size = num_node_in_sector * node_byte_size;

            /*
            sector_byte_sizeは、maxなのか、実際に梱包した時のサイズなのかをはっきりさせる。
            */

            let storage = if !skip_merge_index {
                Storage::new_with_empty_file(
                    &unordered_graph_storage_path,
                    file_byte_size.clone() as u64,
                    sector_byte_size,
                )
                .unwrap()
            } else {
                Storage::load(&unordered_graph_storage_path, sector_byte_size).unwrap()
            };
            let mut graph_on_stroage = GraphStore::new(
                vector_reader.get_num_vectors(),
                vector_reader.get_vector_dim(),
                70 * 2,
                storage,
            );
            // println!("edges: {:?}", graph_on_stroage.read_node(&132));

            println!("k-menas on disk and sharded indexing");
            let num_clusters: u8 = 16;
            let (cluster_points, reordered_node_ids) = if !skip_merge_index {
                let (cluster_labels, cluster_points) =
                    on_disk_k_means(&vector_reader, &num_clusters, &mut rng);
                let reordered_node_ids = sharded_index(
                    &vector_reader,
                    &mut graph_on_stroage,
                    &num_clusters,
                    &cluster_labels,
                    &num_node_in_sector,
                    seed,
                );

                let result = (cluster_points, reordered_node_ids);

                let json_string = serde_json::to_string(&result)?;
                let mut file = File::create(cluster_points_path)?;
                file.write_all(json_string.as_bytes())?;

                result
            } else {
                println!("skiped");
                let file = File::open(cluster_points_path)?;
                let reader = BufReader::new(file);

                // JSONをデシリアライズ
                let cluster_points_and_reordered_ids = serde_json::from_reader(reader)?;
                cluster_points_and_reordered_ids
            };

            /* diskへの書き込み */
            println!("writing disk");
            let storage = if !skip_make_backlinks {
                println!("Storage::new_with_empty_file");
                Storage::new_with_empty_file(
                    &ordered_graph_storage_path,
                    file_byte_size.clone() as u64,
                    sector_byte_size,
                )
                .unwrap()
            } else {
                println!("skiped");
                Storage::load(&ordered_graph_storage_path, sector_byte_size).unwrap()
            };

            println!("GraphStore::new");
            let ordered_graph_on_storage = GraphStore::new(
                vector_reader.get_num_vectors(),
                vector_reader.get_vector_dim(),
                70 * 2,
                storage,
            );

            println!("reordered_node_ids");
            println!(
                "reordered_node_ids_count {}",
                reordered_node_ids.iter().flatten().count()
            );

            reordered_node_ids
                .iter()
                .enumerate()
                .for_each(|(sector_index, node_ids)| {
                    let mut node_offset = 0;
                    let mut buffer: Vec<u8> = vec![0; sector_byte_size];
                    for node_id in node_ids {
                        let serialized_node =
                            graph_on_stroage.read_serialized_node(&(*node_id as usize));
                        let node_offset_end = node_offset + node_byte_size;
                        buffer[node_offset..node_offset_end].copy_from_slice(&serialized_node);
                        node_offset = node_offset_end;
                    }

                    ordered_graph_on_storage
                        .write_serialized_sector(&sector_index, &buffer)
                        .unwrap();
                });

            /* node_index とstore_indexの変換表を作る。 */
            println!("making node_index_to_store_index");
            let node_index_to_store_index: Vec<u32> = reordered_node_ids
                .into_iter()
                .flatten()
                .enumerate()
                .map(|(store_index, node_index)| (node_index, store_index))
                .sorted()
                .map(|(_, store_index)| store_index as u32)
                .collect();

            let ave_point: Point = Point::from_f32_vec(
                cluster_points
                    .iter()
                    .map(|(p, _)| p)
                    .fold(
                        Point::from_f32_vec(vec![0.0; vector_reader.get_vector_dim()]),
                        |acc, x| acc.add(x),
                    )
                    .to_f32_vec()
                    .into_iter()
                    .map(|x| x / cluster_points.len() as f32)
                    .collect(),
            );

            // ave_pointの最近傍ノードを探すために、ave_pointが属するクラスターを探す。
            // WIP: sharded_indexの方からそれぞれのシャードのcentoridを持ってくればいい？
            let tmp_start_index = cluster_points
                .iter()
                .map(|(p, v)| (ave_point.distance(&p), v))
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less))
                .unwrap()
                .1
                .clone();

            let mut graph = Graph::new(
                ordered_graph_on_storage,
                node_index_to_store_index.clone(),
                tmp_start_index as u32,
            );

            let (dist, centroid_node_index) = vectune::search(&mut graph, &ave_point, 10).0[0];

            println!(
                "centroid_node_index: {}. dist: {}",
                centroid_node_index, dist
            );

            graph.set_start_node_index(centroid_node_index);

            /* test search */

            let random_vector = Point::from_f32_vec(
                (0..vector_reader.get_vector_dim())
                    .map(|_| rng.gen::<f32>())
                    .collect(),
            );
            let (dist, index) = (0..vector_reader.get_num_vectors())
                .into_par_iter()
                .map(|index| {
                    let (vector, _) = graph_on_stroage.read_node(&index).unwrap();
                    (random_vector.distance(&Point::from_f32_vec(vector)), index)
                })
                .reduce_with(|acc, x| if acc.0 < x.0 { acc } else { x })
                .unwrap();
            println!("random_vector truth: {} {}", index, dist);

            let k_ann = vectune::search(&mut graph, &random_vector, 10).0;
            println!("results: {:?}", k_ann);

            let data = (node_index_to_store_index, centroid_node_index);
            let json_string = serde_json::to_string(&data)?;
            let mut file = File::create(graph_json_path)?;
            file.write_all(json_string.as_bytes())?;

            /* recall-rate */

            let query_vector_reader = OriginalVectorReader::new(&query_vector_path)?;
            let groundtruth_reader = OriginalVectorReader::new(&groundtruth_path)?;

            let progress = Some(ProgressBar::new(1000));
            let progress_done = AtomicUsize::new(0);
            if let Some(bar) = &progress {
                bar.set_length(query_vector_reader.get_num_vectors() as u64);
                bar.set_message("Gordering");
            }

            // let query_iter = query_vector_reader.get_num_vectors();
            let query_iter = 20;

            let hit = (0..query_iter)
                .into_iter()
                .map(|query_index| {
                    let query_vector: Vec<f32> = query_vector_reader.read(&query_index).unwrap();
                    let groundtruth_indexies: Vec<u32> =
                        groundtruth_reader.read(&query_index).unwrap();

                    let groundtruth_vectors: Vec<Vec<f32>> = groundtruth_indexies.iter().map(|i| vector_reader.read(&(*i as usize)).unwrap()).collect();
                    
                    

                    let k_ann =
                        vectune::search(&mut graph, &Point::from_f32_vec(query_vector), 5).0;

                    let result_top_5: Vec<u32> = k_ann.into_iter().map(|(_, i)| i).collect();
                    let top5_groundtruth = &groundtruth_indexies[0..5];
                    println!("{:?}\n{:?}\n\n", top5_groundtruth, result_top_5);
                    let mut hit = 0;
                    for res in result_top_5 {
                        if top5_groundtruth.contains(&res) {
                            hit += 1;
                        }
                    }
                    if let Some(bar) = &progress {
                        let value = progress_done.fetch_add(1, atomic::Ordering::Relaxed);
                        if value % 1000 == 0 {
                            bar.set_position(value as u64);
                        }
                    }

                    hit
                })
                .reduce(|acc, x| acc + x)
                .unwrap();

            if let Some(bar) = &progress {
                bar.finish();
            }

            println!("5-recall-rate@5: {}", hit as f32 / (5 * query_iter) as f32);
        }

        Commands::Search => {
            search();
        }
    }

    Ok(())
}

fn search() {
    println!("Running search...");
}

/*

pruneには、個数を制限するロジックはついていない？

todo: external sortのメモリサイズの指定を

cacheは、RefCellを使っているので、thread local出ないといけない

*/
