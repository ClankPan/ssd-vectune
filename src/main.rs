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
    io::{BufReader, Read, Write},
    sync::atomic::{self, AtomicUsize},
    time::Instant,
};

use anyhow::Result;
// use bit_vec::BitVec;
use bytesize::KB;
// use cache::Cache;
// use ext_sort::{buffer::LimitedBufferBuilder, ExternalSorter, ExternalSorterBuilder};
use graph::{Graph, UnorderedGraph};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use k_means::on_disk_k_means;
// use node_reader::{EdgesIterator, GraphOnStorage, GraphOnStorageTrait};
use original_vector_reader::{OriginalVectorReader, OriginalVectorReaderTrait};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use sharded_index::sharded_index;
// use storage::StorageTrait;
// use single_index::single_index;
use vectune::{GraphInterface, PointInterface};

use crate::{graph_store::GraphStore, point::Point, storage::Storage};

// type VectorIndex = usize;

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
        skip_k_means: bool,

        #[arg(long)]
        skip_merge_index: bool,

        #[arg(long)]
        skip_make_backlinks: bool,

        #[arg(long)]
        skip_test_recall_rate: bool,

        original_vector_path: String,

        destination_directory: String,

        max_sector_k_byte_size: usize,

        dataset_size: usize,

        max_chunk_giga_byte_size: u64,

        num_clusters: u8,
    },
    Debug {
        graph_metadata_path: String,
    },
    UnorderRecallRate {
        query_path: String,
        original_vector_path: String,
        ground_truth_path: String,
        unordered_graph_storage_path: String,
        graph_metadata_path: String,
        edge_degrees: usize,
        size_l: usize,
    },
    /// Executes the search process
    RecallRate {
        query_path: String,
        ground_truth_path: String,
        ordered_graph_storage_path: String,
        graph_metadata_path: String,
    },
    Prune {
        unordered_graph_storage_path: String,
        graph_metadata_path: String,
        new_edge_degrees: usize,
    },
}

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct GraphMetadata {
    node_index_to_store_index: Vec<u32>,
    medoid_node_index: u32,
    sector_byte_size: usize,
    num_vectors: usize,
    vector_dim: usize,
    edge_degrees: usize,
}

impl GraphMetadata {
    pub fn new(
        node_index_to_store_index: Vec<u32>,
        medoid_node_index: u32,
        sector_byte_size: usize,
        num_vectors: usize,
        vector_dim: usize,
        edge_degrees: usize,
    ) -> Self {
        Self {
            node_index_to_store_index,
            medoid_node_index,
            sector_byte_size,
            num_vectors,
            vector_dim,
            edge_degrees,
        }
    }

    pub fn load(path: &str) -> Result<Self> {
        let mut file = File::open(path).expect("file not found");

        // ファイルの内容を読み込む
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("something went wrong reading the file");

        // JSONデータをPerson構造体にデシリアライズ
        let metadata: Self = serde_json::from_str(&contents)?;
        Ok(metadata)
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let json_string = serde_json::to_string(self)?;
        let mut file = File::create(path)?;
        file.write_all(json_string.as_bytes())?;
        Ok(())
    }

    pub fn load_debug(path: &str, sector_byte_size: usize) -> Result<Self> {
        let mut file = File::open(path).expect("file not found");

        // ファイルの内容を読み込む
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("something went wrong reading the file");

        // JSONデータをPerson構造体にデシリアライズ
        let (node_index_to_store_index, medoid_node_index): (Vec<u32>, u32) =
            serde_json::from_str(&contents)?;
        let metadata = Self {
            node_index_to_store_index,
            medoid_node_index,
            sector_byte_size,
            num_vectors: 100 * 1000000,
            vector_dim: 96,
            edge_degrees: 70 * 2,
        };

        metadata.save(&format!("{path}.debug"))?;

        Ok(metadata)
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let seed = 01234; // wip
    let mut rng = SmallRng::seed_from_u64(seed);

    match cli.command {
        Commands::Build {
            skip_k_means,
            skip_merge_index,
            skip_make_backlinks, // wip
            skip_test_recall_rate,
            original_vector_path,
            destination_directory,
            max_sector_k_byte_size,
            dataset_size,
            max_chunk_giga_byte_size,
            num_clusters,
        } => {
            let max_sector_byte_size = max_sector_k_byte_size * KB as usize;

            /* file path */
            let unordered_graph_storage_path =
                format!("{destination_directory}/unordered_graph.graph");
            let ordered_graph_storage_path = format!("{destination_directory}/ordered_graph.graph");
            let graph_json_path = format!("{destination_directory}/graph.json");
            let cluster_labels_and_point_path =
                format!("{destination_directory}/cluster_labels_and_point.json");
            let reordered_node_ids_path =
                format!("{destination_directory}/reordered_node_ids.json");

            /* read original vectors */
            println!("reading vector file");
            // 10Mだと大きすぎるので、小さなデータセットをここから作る。
            // #[cfg(not(feature = "size"))]
            // let vector_reader = OriginalVectorReader::new(&original_vector_path)?;
            let mut vector_reader = {
                println!("dataset_size: {} million", dataset_size);
                OriginalVectorReader::new_with(&original_vector_path, dataset_size)?
            };

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

            println!("k-menas on disk");
            // let num_clusters: u8 = 16;
            let (cluster_labels, cluster_points) = if !skip_k_means {
                let (cluster_labels, cluster_points) = on_disk_k_means(
                    &mut vector_reader,
                    &num_clusters,
                    max_chunk_giga_byte_size,
                    &mut rng,
                );

                let json_string =
                    serde_json::to_string(&(cluster_labels.clone(), cluster_points.clone()))?;
                let mut file = File::create(cluster_labels_and_point_path)?;
                file.write_all(json_string.as_bytes())?;

                (cluster_labels, cluster_points)
            } else {
                println!("skiped");
                let file = File::open(cluster_labels_and_point_path)?;
                let reader = BufReader::new(file);

                // JSONをデシリアライズ
                serde_json::from_reader(reader)?
            };

            println!("sharded indexing");
            let reordered_node_ids = if !skip_merge_index {
                let reordered_node_ids = sharded_index(
                    &vector_reader,
                    &mut graph_on_stroage,
                    &num_clusters,
                    &cluster_labels,
                    &num_node_in_sector,
                    seed,
                );

                let json_string = serde_json::to_string(&reordered_node_ids)?;
                let mut file = File::create(reordered_node_ids_path)?;
                file.write_all(json_string.as_bytes())?;

                reordered_node_ids
            } else {
                println!("skiped");
                let file = File::open(reordered_node_ids_path)?;
                let reader = BufReader::new(file);

                // JSONをデシリアライズ
                let reordered_ids = serde_json::from_reader(reader)?;
                reordered_ids
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
                reordered_node_ids.par_iter().flatten().count()
            );

            reordered_node_ids
                .par_iter()
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
                    .into_par_iter()
                    .map(|x| x / cluster_points.len() as f32)
                    .collect(),
            );

            // ave_pointの最近傍ノードを探すために、ave_pointが属するクラスターを探す。
            // WIP: sharded_indexの方からそれぞれのシャードのcentoridを持ってくればいい？
            let tmp_start_index = cluster_points
                .par_iter()
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

            let data = GraphMetadata::new(
                node_index_to_store_index,
                centroid_node_index,
                sector_byte_size,
                vector_reader.get_num_vectors(),
                vector_reader.get_vector_dim(),
                70 * 2,
            );
            let json_string = serde_json::to_string(&data)?;
            let mut file = File::create(graph_json_path)?;
            file.write_all(json_string.as_bytes())?;

            /* test search */

            if !skip_test_recall_rate {
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

                /* recall-rate */

                let query_vector_reader =
                    OriginalVectorReader::new("test_vectors/query.public.10K.fbin")?;
                let groundtruth: Vec<Vec<u32>> =
                    read_ivecs("test_vectors/gt/deep10M_groundtruth.ivecs").unwrap();

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
                        let query_vector: Vec<f32> =
                            query_vector_reader.read(&query_index).unwrap();
                        // let groundtruth_indexies: Vec<u32> =
                        //     groundtruth_reader.read(&query_index).unwrap();

                        // let groundtruth_vectors: Vec<Vec<f32>> = ivecs_groundtruth[query_index].iter().map(|i| vector_reader.read(&(*i as usize)).unwrap()).collect();

                        let k_ann =
                            vectune::search(&mut graph, &Point::from_f32_vec(query_vector), 5).0;

                        let result_top_5: Vec<u32> = k_ann.into_iter().map(|(_, i)| i).collect();
                        let top5_groundtruth = &groundtruth[query_index][0..5];
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
        }

        Commands::Debug {
            graph_metadata_path,
        } => {
            let _graph_metadata = GraphMetadata::load_debug(&graph_metadata_path, 3792).unwrap();
        }

        Commands::UnorderRecallRate {
            query_path,
            original_vector_path,
            ground_truth_path,
            unordered_graph_storage_path,
            graph_metadata_path,
            edge_degrees,
            size_l,
        } => {
            let graph_metadata = GraphMetadata::load(&graph_metadata_path).unwrap();

            let max_sector_byte_size = 4 * KB as usize;
            let node_byte_size = (graph_metadata.vector_dim * 4 + 140 * 4 + 4) as usize;
            // let file_byte_size = graph_metadata.num_vectors * node_byte_size;
            let num_node_in_sector = max_sector_byte_size / node_byte_size;
            let sector_byte_size = num_node_in_sector * node_byte_size;
            println!("sector_byte_size {}", sector_byte_size);
            let _num_sectors = (graph_metadata.num_vectors * node_byte_size + sector_byte_size - 1)
                / sector_byte_size;

            let storage = Storage::load(
                &unordered_graph_storage_path,
                graph_metadata.sector_byte_size,
            )
            .unwrap();

            let unordered_graph_on_storage = GraphStore::new(
                graph_metadata.num_vectors,
                graph_metadata.vector_dim,
                edge_degrees,
                storage,
                // cache,
            );

            let mut graph = UnorderedGraph::new(
                unordered_graph_on_storage,
                graph_metadata.medoid_node_index as u32,
            );

            graph.set_size_l(size_l);

            let query_vector_reader = OriginalVectorReader::new(&query_path)?;
            // let original_vector_reader = OriginalVectorReader::new(&original_vector_path)?;
            let groundtruth: Vec<Vec<u32>> = read_ivecs(&ground_truth_path).unwrap();

            let query_iter = query_vector_reader.get_num_vectors();

            let mut total_time = 0;

            let mut rng = SmallRng::seed_from_u64(rand::random());

            // let mut scores = vec![0 as u32; graph_metadata.num_vectors];

            let hit = (0..query_iter)
                .into_iter()
                .map(|query_index| {
                    // let query_index = rng.gen_range(0..query_vector_reader.get_num_vectors());
                    let query_vector: Vec<f32> = query_vector_reader.read(&query_index).unwrap();
                    println!("query_index {query_index}");

                    // let query_vector: Vec<f32> = vec![rng.gen::<f32>(); 96];

                    let start = Instant::now();
                    let (k_ann, visited) =
                        vectune::search(&mut graph, &Point::from_f32_vec(query_vector), 5);
                    let t = start.elapsed().as_millis();
                    println!("{t} milli sec, visited len: {}", visited.len());
                    total_time += t;

                    let result_top_5: Vec<u32> = k_ann.into_iter().map(|(_, i)| i).collect();
                    let top5_groundtruth = &groundtruth[query_index][0..5];
                    // println!("{:?}\n{:?}\n\n", top5_groundtruth, result_top_5);
                    let mut hit = 0;
                    for res in result_top_5 {
                        if top5_groundtruth.contains(&res) {
                            hit += 1;
                        }
                    }

                    // visited
                    //     .into_iter()
                    //     .for_each(|(_, index)| scores[index as usize] += 1);

                    hit
                })
                .reduce(|acc, x| acc + x)
                .unwrap();

            println!("5-recall-rate@5: {}", hit as f32 / (5 * query_iter) as f32);
            println!("average search time: {}", total_time / query_iter as u128);


            // let mut scores: Vec<_> = scores.into_iter().enumerate().collect();
            // scores.sort_by(|a, b| b.1.cmp(&a.1));
            // scores.truncate(2_886_402);
            // // scores.truncate(360_800);
            // scores.sort_by(|a, b| a.0.cmp(&b.0));
            // let scores: Vec<_> = scores.into_par_iter().map(|(i, _)| i).collect();


            // let query_iter = 1000;

            // let (hit, total_visited) = (0..query_iter)
            //     .into_iter()
            //     .map(|query_count| {
            //         let query_index = rng.gen_range(10 * 1000000..original_vector_reader.get_num_vectors());
            //         let query_vector: Vec<f32> = original_vector_reader.read(&query_index).unwrap();
            //         // let query_vector: Vec<f32> = query_vector_reader.read(&query_index).unwrap();

            //         // let query_vector: Vec<f32> = vec![rng.gen::<f32>(); 96];

            //         let start = Instant::now();
            //         let (_k_ann, visited) =
            //             vectune::search(&mut graph, &Point::from_f32_vec(query_vector), 5);
            //         let t = start.elapsed().as_millis();
            //         println!("query_count {query_count}, {t} milli sec, visited len: {}", visited.len());
            //         // total_time += t;

            //         let visited_len = visited.len();
            //         let hit_in_visited = visited
            //             .into_iter()
            //             .filter(|(_, index)| scores.contains(&(*index as usize))).count();

            //         // println!("{}", hit_in_visited as f32 / visited_len as f32);

            //         (hit_in_visited, visited_len)

            //     })
            //     .reduce(|acc, x| (acc.0 + x.0, acc.1 + x.1))
            //     .unwrap();

            // println!("{}%", (hit as f32 / total_visited as f32) * 100.0)
        }

        Commands::RecallRate {
            query_path,
            ground_truth_path,
            ordered_graph_storage_path,
            graph_metadata_path,
        } => {
            /* recall-rate */

            let graph_metadata = GraphMetadata::load(&graph_metadata_path).unwrap();

            let max_sector_byte_size = 4 * KB as usize;
            let node_byte_size = (graph_metadata.vector_dim * 4 + 140 * 4 + 4) as usize;
            // let file_byte_size = graph_metadata.num_vectors * node_byte_size;
            let num_node_in_sector = max_sector_byte_size / node_byte_size;
            let sector_byte_size = num_node_in_sector * node_byte_size;
            println!("sector_byte_size {}", sector_byte_size);
            let _num_sectors = (graph_metadata.num_vectors * node_byte_size + sector_byte_size - 1)
                / sector_byte_size;

            let storage =
                Storage::load(&ordered_graph_storage_path, graph_metadata.sector_byte_size)
                    .unwrap();

            // let cache = Cache::new(num_sectors, 96, 70 * 2, storage);

            let ordered_graph_on_storage = GraphStore::new(
                graph_metadata.num_vectors,
                graph_metadata.vector_dim,
                graph_metadata.edge_degrees,
                storage,
                // cache,
            );

            let mut graph = Graph::new(
                ordered_graph_on_storage,
                graph_metadata.node_index_to_store_index.clone(),
                graph_metadata.medoid_node_index as u32,
            );

            let query_vector_reader = OriginalVectorReader::new(&query_path)?;
            let groundtruth: Vec<Vec<u32>> = read_ivecs(&ground_truth_path).unwrap();

            // let query_iter = query_vector_reader.get_num_vectors();
            let query_iter = 100;

            let mut total_time = 0;

            let hit = (0..query_iter)
                .into_iter()
                .map(|query_index| {
                    let query_vector: Vec<f32> = query_vector_reader.read(&query_index).unwrap();
                    // let groundtruth_indexies: Vec<u32> =
                    //     groundtruth_reader.read(&query_index).unwrap();

                    // let groundtruth_vectors: Vec<Vec<f32>> = ivecs_groundtruth[query_index].iter().map(|i| vector_reader.read(&(*i as usize)).unwrap()).collect();

                    let start = Instant::now();
                    let (k_ann, visited) =
                        vectune::search(&mut graph, &Point::from_f32_vec(query_vector), 5);
                    let t = start.elapsed().as_millis();
                    println!("{t} milli sec, visited len: {}", visited.len());
                    total_time += t;

                    let result_top_5: Vec<u32> = k_ann.into_iter().map(|(_, i)| i).collect();
                    let top5_groundtruth = &groundtruth[query_index][0..5];
                    println!("{:?}\n{:?}\n\n", top5_groundtruth, result_top_5);
                    let mut hit = 0;
                    for res in result_top_5 {
                        if top5_groundtruth.contains(&res) {
                            hit += 1;
                        }
                    }

                    hit
                })
                .reduce(|acc, x| acc + x)
                .unwrap();

            println!("5-recall-rate@5: {}", hit as f32 / (5 * query_iter) as f32);
            println!("average search time: {}", total_time / query_iter as u128);
        }

        Commands::Prune {
            unordered_graph_storage_path,
            graph_metadata_path,
            new_edge_degrees,
        } => {
            let style = ProgressStyle::default_bar()
            .template(
                "{spinner:.green}  {msg}\n[{elapsed_precise}] {percent:>3}% {wide_bar:.cyan/blue}",
            )
            .unwrap();

            // pointを含めてedgesをtupleのvecでストレージから取り込む
            let (sector_byte_size, num_vectors, vector_dim, edge_degrees) = {
                let graph_metadata: GraphMetadata =
                    GraphMetadata::load(&graph_metadata_path).unwrap();
                (
                    graph_metadata.sector_byte_size,
                    graph_metadata.num_vectors,
                    graph_metadata.vector_dim,
                    graph_metadata.edge_degrees,
                )
            };

            let storage = Storage::load(&unordered_graph_storage_path, sector_byte_size).unwrap();
            let unordered_graph_on_storage =
                GraphStore::new(num_vectors, vector_dim, edge_degrees, storage);

            let pb = ProgressBar::new(num_vectors as u64).with_style(style.clone());
            pb.set_message("reading node form disk");
            let progress_done = AtomicUsize::new(0);

            let nodes: Vec<(Point, Vec<u32>)> = (0..num_vectors)
                .into_par_iter()
                .map(|store_index| {
                    let (vector, edges) =
                        unordered_graph_on_storage.read_node(&store_index).unwrap();

                    let value = progress_done.fetch_add(1, atomic::Ordering::Relaxed);
                    if value % 1000 == 0 {
                        pb.set_position(value as u64);
                    }

                    (Point::from_f32_vec(vector), edges)
                })
                .collect();

            let (points, all_edges): (Vec<_>, Vec<_>) = nodes.into_par_iter().unzip();

            pb.finish();

            // 全てに対して刈り取る
            let pb = ProgressBar::new(num_vectors as u64).with_style(style.clone());
            pb.set_message("pruning");
            let progress_done = AtomicUsize::new(0);
            let pruned: Vec<Vec<u32>> = all_edges
                .into_par_iter()
                .enumerate()
                .map(|(index, edges)| {
                    let p = &points[index];
                    let candidates: Vec<(f32, u32)> = edges
                        .into_iter()
                        .map(|edge_index| (points[edge_index as usize].distance(p), edge_index))
                        .collect();
                    // let mut candidates: Vec<(f32, u32)> = Vec::with_capacity(edges.len());
                    // for edge_index in edges {
                    //     candidates.push((points[edge_index as usize].distance(p), edge_index));
                    // }

                    let new_edges = prune::<Point>(&points, candidates, &new_edge_degrees, &2.0);

                    let value = progress_done.fetch_add(1, atomic::Ordering::Relaxed);
                    if value % 1000 == 0 {
                        pb.set_position(value as u64);
                    }

                    new_edges
                })
                .collect();

            pb.finish();

            // 書き込み先のstorageを用意
            let pb = ProgressBar::new(num_vectors as u64).with_style(style.clone());
            pb.set_message("writing into disk");
            let progress_done = AtomicUsize::new(0);
            let node_byte_size = (vector_dim * 4 + 140 * 4 + 4) as usize;
            let file_byte_size = num_vectors * node_byte_size;
            let pruned_unordered_graph_on_storage = GraphStore::new(
                num_vectors,
                vector_dim,
                new_edge_degrees,
                Storage::new_with_empty_file(
                    &format!("{unordered_graph_storage_path}.{new_edge_degrees}_degrees"),
                    file_byte_size as u64,
                    sector_byte_size,
                )
                .unwrap(),
            );

            points
                .into_par_iter()
                .zip(pruned)
                .enumerate()
                .for_each(|(index, (points, edges))| {
                    let vector = points.to_f32_vec();
                    pruned_unordered_graph_on_storage
                        .write_node(&index, &vector, &edges)
                        .unwrap();

                    let value = progress_done.fetch_add(1, atomic::Ordering::Relaxed);
                    if value % 1000 == 0 {
                        pb.set_position(value as u64);
                    }
                });

            pb.finish();
        }
    }

    Ok(())
}

fn prune<P>(
    points: &Vec<Point>,
    mut candidates: Vec<(f32, u32)>,
    builder_r: &usize,
    builder_a: &f32,
) -> Vec<u32>
where
    P: PointInterface,
{
    let mut new_n_out = vec![];

    while let Some((first, rest)) = candidates.split_first() {
        let (_, pa) = *first; // pa is p asterisk (p*), which is nearest point to p in this loop
        new_n_out.push(pa);

        if new_n_out.len() == *builder_r {
            break;
        }
        candidates = rest.to_vec();

        // if α · d(p*, p') <= d(p, p') then remove p' from v
        candidates.retain(|&(dist_xp_pd, pd)| {
            // let pa_point = &self.nodes[pa].p;
            // let pd_point = &self.nodes[pd].p;
            let pa_point = &points[pa as usize]; //get(&pa);
            let pd_point = &points[pd as usize]; //get(&pd);
            let dist_pa_pd = pa_point.distance(pd_point);

            builder_a * dist_pa_pd > dist_xp_pd
        })
    }

    new_n_out
}

use byteorder::{LittleEndian, ReadBytesExt};

fn read_ivecs(file_path: &str) -> std::io::Result<Vec<Vec<u32>>> {
    let file = File::open(file_path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();

    while let Ok(dim) = reader.read_i32::<LittleEndian>() {
        let mut vec = Vec::with_capacity(dim as usize);
        for _ in 0..dim {
            let val = reader.read_i32::<LittleEndian>()?;
            vec.push(val);
        }
        vectors.push(vec);
    }

    Ok(vectors
        .into_iter()
        .map(|gt| gt.into_iter().map(|g| g as u32).collect())
        .collect())
}

fn _read_ibin(file_path: &str) -> Result<Vec<Vec<f32>>> {
    let data = std::fs::read(file_path)?;

    let num_vectors = u32::from_le_bytes(data[0..4].try_into()?) as usize;
    let vector_dim = u32::from_le_bytes(data[4..8].try_into()?) as usize;
    let start_offset = 8;
    let vector_byte_size = vector_dim * 4;

    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .into_iter()
        .map(|i| {
            let offset = start_offset + vector_byte_size * i;

            let vector: &[f32] =
                bytemuck::try_cast_slice(&data[offset..offset + vector_byte_size]).unwrap();
            vector.to_vec()
        })
        .collect();

    Ok(vectors)
}

/*

pruneには、個数を制限するロジックはついていない？

todo: external sortのメモリサイズの指定を

cacheは、RefCellを使っているので、thread local出ないといけない

*/
