#![cfg_attr(feature = "simd", feature(portable_simd))]

#[cfg(test)]
mod tests;

pub mod cache;
pub mod graph;
pub mod graph_store;
pub mod k_means;
pub mod original_vector_reader;
pub mod point;
pub mod sharded_index;
pub mod single_index;
pub mod storage;
pub mod utils;

use std::{
    fs::File,
    io::{BufReader, Write},
    path::{self, Path, PathBuf},
};

use anyhow::Result;
use bit_vec::BitVec;
use bytesize::{GB, KB};
use ext_sort::{buffer::LimitedBufferBuilder, ExternalSorter, ExternalSorterBuilder};
use graph::Graph;
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};

use k_means::on_disk_k_means;
// use node_reader::{EdgesIterator, GraphOnStorage, GraphOnStorageTrait};
use original_vector_reader::{OriginalVectorReader, OriginalVectorReaderTrait};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use sharded_index::sharded_index;
// use single_index::single_index;
use vectune::PointInterface;

use crate::{
    graph_store::{EdgesIterator, GraphStore},
    point::Point,
    storage::Storage,
};

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
            skip_make_backlinks,
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
                backlinks_path,
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
                    directory.join("query.public.10K.fbin").display().to_string(),
                    directory.join("groundtruth.public.10K.ibin").display().to_string(),
                    directory.join("backlinks.json").display().to_string(),
                )
            } else {
                panic!()
            };

            /* read original vectors */
            println!("reading vector file");
            // 10Mだと大きすぎるので、小さなデータセットをここから作る。
            let vector_reader = OriginalVectorReader::new(&original_vector_path)?;

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
            let cluster_points = if !skip_merge_index {
                let (cluster_labels, cluster_points) =
                    on_disk_k_means(&vector_reader, &num_clusters, &mut rng);
                sharded_index(
                    &vector_reader,
                    &mut graph_on_stroage,
                    &num_clusters,
                    &cluster_labels,
                    seed,
                );

                let json_string = serde_json::to_string(&cluster_points)?;
                let mut file = File::create(cluster_points_path)?;
                file.write_all(json_string.as_bytes())?;

                cluster_points
            } else {
                println!("skiped");
                let file = File::open(cluster_points_path)?;
                let reader = BufReader::new(file);

                // JSONをデシリアライズ
                let cluster_points = serde_json::from_reader(reader)?;
                cluster_points
            };

            /* gordering */
            // 2. 並び替えの順番をもとに、ssdに書き込む。

            // debug
            println!("vector len {}", vector_reader.get_num_vectors());

            // WIP: buffer size
            let sort_buffer_size = 5 * GB as usize;

            println!("gordering");
            println!("make backlinks");
            let backlinks = if !skip_make_backlinks {
                let edge_iter = EdgesIterator::new(&graph_on_stroage);

                println!("make sorter");
                let sorter: ExternalSorter<(u32, u32), std::io::Error, LimitedBufferBuilder> =
                    ExternalSorterBuilder::new()
                        .with_tmp_dir(path::Path::new("./"))
                        .with_buffer(LimitedBufferBuilder::new(sort_buffer_size, false))
                        .build()
                        .unwrap();
                println!("sort edges");
                let sorted = sorter.sort_by(edge_iter, |a, b| a.0.cmp(&b.0)).unwrap();
                println!("make backlinks");
    
                let backlinks: Vec<Vec<u32>> = sorted
                    .into_iter()
                    .map(Result::unwrap)
                    .chunk_by(|&(key, _)| key)
                    .into_iter()
                    .map(|(_key, group)| {
                        // println!("{}", _key);
                        group.map(|(_, edge)| edge).collect()
                    })
                    .collect();


                let json_string = serde_json::to_string(&backlinks)?;
                let mut file = File::create(backlinks_path)?;
                file.write_all(json_string.as_bytes())?;

                backlinks
            } else {
                println!("skiped");
                let file = File::open(backlinks_path)?;
                let reader = BufReader::new(file);

                // JSONをデシリアライズ
                let backlinks: Vec<Vec<u32>> = serde_json::from_reader(reader)?;
                backlinks
            };


            println!("backlinks.len(): {}", backlinks.len());


            println!("get_backlinks clousre");
            let get_backlinks = |id: &u32| -> Vec<u32> { backlinks[*id as usize].clone() };

            println!("get_edges clousre");
            let get_edges =
                |id: &u32| -> Vec<u32> { graph_on_stroage.read_edges(&(*id as usize)).unwrap() };

            println!("target_node_bit_vec");
            let target_node_bit_vec = BitVec::from_elem(vector_reader.get_num_vectors(), true);
            let window_size = num_node_in_sector;
            println!("do vectune::gorder");
            let reordered_node_ids = vectune::gorder(
                get_edges,
                get_backlinks,
                target_node_bit_vec,
                window_size,
                &mut rng,
            );

            /* diskへの書き込み */
            println!("writing disk");
            let storage = if true {
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

            /*
            WIP: 注意

            sector_packingは、最後の一個を別でやらないと、個数が足らないsectorが、途中に挟まってしまう。
            
            
            */

            println!("reordered_node_ids");
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
            // reordered_node_ids
            //     .chunks(window_size)
            //     .enumerate()
            //     .par_bridge()
            //     .for_each(|(sector_index, node_ids)| {
            //         let mut node_offset = 0;
            //         let mut buffer: Vec<u8> = vec![0; sector_byte_size];
            //         for node_id in node_ids {
            //             let serialized_node =
            //                 graph_on_stroage.read_serialized_node(&(*node_id as usize));
            //             let node_offset_end = node_offset + node_byte_size;
            //             buffer[node_offset..node_offset_end].copy_from_slice(&serialized_node);
            //             node_offset = node_offset_end;
            //         }

            //         ordered_graph_on_storage
            //             .write_serialized_sector(&sector_index, &buffer)
            //             .unwrap();
            //     });

            /* node_index とstore_indexの変換表を作る。 */
            let node_index_to_store_index: Vec<u32> = reordered_node_ids
                .into_iter()
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

            /* test recall-rate */

            let query_vector_reader = OriginalVectorReader::new(&query_vector_path)?;
            


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
