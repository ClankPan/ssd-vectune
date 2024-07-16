#![cfg_attr(feature = "simd", feature(portable_simd))]

#[cfg(test)]
mod tests;

// pub mod cache;
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
    collections::HashMap,
};

use anyhow::Result;
// use bit_vec::BitVec;
use bytesize::KB;
// use cache::Cache;
// use ext_sort::{buffer::LimitedBufferBuilder, ExternalSorter, ExternalSorterBuilder};
use graph::{Graph, UnorderedGraph, GraphMetadata};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
// use ndarray::{Array2, ArrayView2, Axis};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use k_means::{on_disk_k_means, on_disk_k_means_pq};
// use node_reader::{EdgesIterator, GraphOnStorage, GraphOnStorageTrait};
use original_vector_reader::{read_ivecs, OriginalVectorReader, OriginalVectorReaderTrait};
use rand::{rngs::SmallRng, thread_rng, Rng, SeedableRng};
// use rustc_hash::FxHashMap;
use sharded_index::sharded_index;
// use storage::StorageTrait;
// use single_index::single_index;
use vectune::PointInterface;

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
        pq_table_path: String,
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
    Pq {
        original_vector_path: String,
        destination_directory: String,
        max_chunk_giga_byte_size: usize,
        dataset_size_million: usize,
    },
    TestPq {
        original_vector_path: String,
        pq_path: String,
    },
    TestKdTree {
        unordered_graph_storage_path: String,
        graph_metadata_path: String,
        edge_degrees: usize,
    },

    BuildAnnoyForest {
        unordered_graph_storage_path: String,
        graph_metadata_path: String,
        edge_degrees: usize,
    }
}

use serde::{Deserialize, Serialize};










fn _encode_24bits(v: [u8; 8]) -> [u8; 3] {
    let u3_0 = v[0] << 5;
    let u3_1 = v[1] << 2;
    let u3_2_1: u8 = v[2] >> 1;

    let u3_a = u3_0 & u3_1 & u3_2_1;

    let u3_2_2 = v[2] << 7;
    let u3_3 = v[3] << 4;
    let u3_4 = v[4] << 1;
    let u3_5_1 = v[5] >> 2;

    let u3_b = u3_2_2 & u3_3 & u3_4 & u3_5_1;

    let u3_5_2 = v[5] << 6;
    let u3_6 = v[6] << 3;
    let u3_7 = v[7];

    let u3_c = u3_5_2 & u3_6 & u3_7;

    [u3_a, u3_b, u3_c]
}

fn _decode(v: [u8; 3]) -> [u8; 8] {
    let u8_0 = v[0] >> 5;
    let u8_1 = (v[0] & 0b00011100) >> 2;
    let u8_2_1 = v[0] & 0b00000011;
    
    let u8_2_2 = v[1] >> 7;
    let u8_3 = (v[1] & 0b01110000) >> 4;
    let u8_4 = (v[1] & 0b00001110) >> 1;
    let u8_5_1 = v[1] & 0b00000001;

    let u8_5_2 = v[2] & 0b11000000 >> 6;
    let u8_6 = (v[2] & 0b00111000) >> 3;
    let u8_7 = v[2] & 0b00000111;

    [u8_0, u8_1, (u8_2_1 & u8_2_2), u8_3, u8_4, (u8_5_1 & u8_5_2), u8_6, u8_7]

}

fn _vec_u3_encoder(vecs: [u8; 32]) -> [u8; 12] {

    let mut encoded = Vec::new();

    encoded.extend(_encode_24bits(vecs[0..8].try_into().unwrap()));
    encoded.extend(_encode_24bits(vecs[8..16].try_into().unwrap()));
    encoded.extend(_encode_24bits(vecs[16..24].try_into().unwrap()));
    encoded.extend(_encode_24bits(vecs[24..32].try_into().unwrap()));

    encoded.try_into().unwrap()
}

fn _vec_u3_decoder(vecs: [u8; 12]) -> [u8; 32] {
    let mut decoded = Vec::new();

    decoded.extend(_decode(vecs[0..3].try_into().unwrap()));
    decoded.extend(_decode(vecs[3..6].try_into().unwrap()));
    decoded.extend(_decode(vecs[6..9].try_into().unwrap()));
    decoded.extend(_decode(vecs[9..12].try_into().unwrap()));

    decoded.try_into().unwrap()
}


enum Tree {
    Node(u8, Box<Tree>, Box<Tree>),
    Leaf(Vec<(Point, u8)>),
}

struct TreeUtils {
    random_splitter_points: Vec<(Point, Point)>,
    max_depth: usize,
}

impl TreeUtils {
    pub fn new(random_splitter_points: Vec<(Point, Point)>, max_depth: usize) -> Self {
        Self {
            random_splitter_points,
            max_depth,
        }
    }

    pub fn _encode(root: Tree) -> ([[u8; 7]; 32], [[u8; 12]; 90]) {
        // depth 0
        let Tree::Node(s0, n0, n1) = root else {panic!()};
        //  
        let Tree::Node(s1, n2, n3) = *n0  else {panic!()};
        let Tree::Node(s2, n4, n5) = *n1  else {panic!()};
        //  
        let Tree::Node(s3, l0, l1) = *n2  else {panic!()};
        let Tree::Node(s4, l2, l3) = *n3  else {panic!()};
        let Tree::Node(s5, l4, l5) = *n4  else {panic!()};
        let Tree::Node(s6, l6, l7) = *n5  else {panic!()};

        let _splitters = [s0, s1, s2, s3, s4, s5, s6];

        let mut _leafs: Vec<(u8, u8)> = vec![];

        let leavs = [*l0, *l1, *l2, *l3, *l4, *l5, *l6, *l7];
        let mut leavs: Vec<(u8, u8)> = leavs.into_iter().enumerate().map(|(leaf_index, leaf)| {
            let Tree::Leaf(points) = leaf else {panic!()};
            points.into_iter().map(|(_point, id)| (id, leaf_index as u8)).collect::<Vec<(u8, u8)>>()
        }).flatten().collect();
        leavs.sort_by(|a, b| a.0.cmp(&b.0)); // sort by id
        let _leaf_indexis: Vec<u8> = leavs.into_iter().map(|(_, leaf_index)| leaf_index).collect();

        todo!()
    }

    pub fn build(&mut self, points: Vec<Point>) -> Tree {
        self.rec_build(points.iter().enumerate().map(|(i, p)| (p, i as u8)).collect(), self.max_depth)
    }


    fn rec_build(&mut self, points: Vec<(&Point, u8)>, depth: usize) -> Tree {

        let mut _rng = thread_rng();

        // if points.len() <= 2 {
        if depth == 1 {
            Tree::Leaf(points.into_iter().map(|(p, i)| (p.clone(), i)).collect())
        } else {

            let (splitter, left, right): (u8, Vec<(&Point, u8)>, Vec<(&Point, u8)>) = self.random_splitter_points.iter().enumerate().map(|(splitter_index, splitter)| {
                let (left, right) = self.random_hyperplane_split(points.clone(), splitter);
                let abs_diff = left.len().abs_diff(right.len());

                (abs_diff, (splitter_index as u8, left, right))
            }).min_by(|(diff_a, _), (diff_b, _)| diff_a.cmp(diff_b)).unwrap().1;

            let left_node = self.rec_build(left, depth-1);
            let right_node = self.rec_build(right, depth-1);

            Tree::Node(splitter.clone(), Box::new(left_node), Box::new(right_node))
        }                    
    }

    fn random_hyperplane_split<'a>(&self, points: Vec<(&'a Point, u8)>, splitter: &(Point, Point)) -> (Vec<(&'a Point, u8)>, Vec<(&'a Point, u8)>) {
        let mut left = vec![];
        let mut right = vec![];
        points.into_iter().for_each(|p| {
            let dist_l = p.0.distance(&splitter.0);
            let dist_r = p.0.distance(&splitter.1);
            if dist_l <= dist_r {
                left.push(p)
            } else {
                right.push(p)
            }
        });

        (left, right)
    }

    pub fn search_tree(&self, tree: Tree, query: &Point, thresh: f32) -> Vec<(Point, u8)> {
        match tree {
            Tree::Node(splitter_index, left, right) => {
                let splitter = &self.random_splitter_points[splitter_index as usize];
                // let splitter = &self.random_select_splitter[splitter_index as usize];
                let dist_l = query.distance(&splitter.0);
                let dist_r = query.distance(&splitter.1);

                if (dist_l - dist_r).abs() < thresh {
                    // println!("close points");
                    let mut vecs = self.search_tree(*left, query, thresh);
                    vecs.extend(self.search_tree(*right, query, thresh));
                    vecs
                
                } else {
                    if dist_l <= dist_r {
                        self.search_tree(*left, query, thresh)
                    } else {
                        self.search_tree(*right, query, thresh)
                    }
                }

            },
            Tree::Leaf(points) => points
        }
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
                            graph_on_stroage.read_serialized_node(node_id);
                        let node_offset_end = node_offset + node_byte_size;
                        buffer[node_offset..node_offset_end].copy_from_slice(&serialized_node);
                        node_offset = node_offset_end;
                    }

                    ordered_graph_on_storage
                        .write_serialized_sector(&(sector_index as u32), &buffer)
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
                        let (vector, _) = graph_on_stroage.read_node(&(index as u32)).unwrap();
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

        Commands::BuildAnnoyForest {
            unordered_graph_storage_path,
            graph_metadata_path,
            edge_degrees,
        } => {

            let graph_metadata = GraphMetadata::load(&graph_metadata_path).unwrap();
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
            );

            let mut rng = thread_rng();

            let num_tree = 70;
            let random_splitter_list: Vec<Vec<(Point, Point)>> = (0..num_tree).into_iter().map(|_| {
                (0..256).map(|_| {
                    let ramdom_index_1 = rng.gen_range(0..graph_metadata.num_vectors) as u32;
                    let ramdom_index_2 = rng.gen_range(0..graph_metadata.num_vectors) as u32;
                    (Point::from_f32_vec(unordered_graph_on_storage
                        .read_node(&ramdom_index_1)
                        .unwrap()
                        .0),
                    Point::from_f32_vec(unordered_graph_on_storage
                        .read_node(&ramdom_index_2)
                        .unwrap()
                        .0))
                }).collect()
            }).collect();


            let _a: Vec<(Vec<u8>, Vec<u8>)> = (0..graph_metadata.num_vectors).into_iter().map(|index| {
                let (vectors, edges) = unordered_graph_on_storage.read_node(&(index as u32)).unwrap();
                let _point = Point::from_f32_vec(vectors);
                let edge_points: Vec<Point> = edges.iter().map(|edge_i| {
                    Point::from_f32_vec(unordered_graph_on_storage.read_node(edge_i).unwrap().0)
                }).collect();

                let _trees: Vec<(Tree, TreeUtils)> = (0..num_tree).map(|iter_count| {
                    let mut tree_utils: TreeUtils = TreeUtils::new(random_splitter_list[iter_count].clone(), 4);
                    let tree = tree_utils.build(edge_points.clone());
                    (tree, tree_utils)
                }).collect();

                todo!()

            }).collect();


            // wip random_splitter_listの保存
            let _random_splitter_list: Vec<Vec<(Vec<f32>, Vec<f32>)>> = random_splitter_list.into_iter().map(|random_splitter| {
                random_splitter.into_iter().map(|(p1, p2)| (p1.to_f32_vec(), p2.to_f32_vec())).collect()
            }).collect();

        }

        Commands::TestKdTree {
            unordered_graph_storage_path,
            graph_metadata_path,
            edge_degrees,
        } => {
            

            let graph_metadata = GraphMetadata::load(&graph_metadata_path).unwrap();
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
            );


            let mut rng = thread_rng();


            let pq: PQ = PQ::load("/Volumes/WD_BLACK/index_deep1b/pq_100M.json")?;
            let codes: [[Point; 256]; 4] = pq
                .cluster_table
                .into_iter()
                .map(|vectors| {
                    vectors
                        .into_iter()
                        .map(|vector| Point::from_f32_vec(vector))
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

            let codebook = pq.pq;

            let sub_dim = Point::DIM / 4;

            let find_code_from_vector = |base_vector: &Vec<f32>| -> [Point; 4] {
                let base_pq: [Point; 4] = (0..4)
                    .into_iter()
                    .zip(codes.iter())
                    .map(|(m, code)| {
                        let sub_point = Point::from_f32_vec(
                            base_vector[m * sub_dim..(m + 1) * sub_dim].to_vec(),
                        );

                        let c = code
                            .iter()
                            .enumerate()
                            .map(|(label, cluster_point)| {
                                (label as u8, cluster_point.distance(&sub_point), cluster_point)
                            })
                            .min_by(|a, b| {
                                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Less)
                            })
                            .unwrap();
                        c.2.clone()
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();

                base_pq
            };

            let find_code = |id: &u32| -> [Point; 4] {

                let pq: [Point; 4] = codebook[*id as usize].iter().zip(codes.iter()).map(|(c, table)| {
                    table[*c as usize].clone()
                }).collect::<Vec<Point>>().try_into().unwrap();
                pq
            };


            // let query_vector_reader = OriginalVectorReader::new(&query_path)?;

            let ramdom_base_index = rng.gen_range(0..graph_metadata.num_vectors) as u32;
            let base_vector = unordered_graph_on_storage
            .read_node(&ramdom_base_index)
            .unwrap()
            .0;
            let base_point = Point::from_f32_vec(
                base_vector.clone()
            );

            let num_iter = 300;

            let base_pq = find_code_from_vector(&base_vector);

            let mut zero_count = 0;
            let top_k = 1;


            let num_tree = 70;
            let random_splitter_list: Vec<Vec<(Point, Point)>> = (0..num_tree).into_iter().map(|_| {
                (0..256).map(|_| {
                    let ramdom_index_1 = rng.gen_range(0..graph_metadata.num_vectors) as u32;
                    let ramdom_index_2 = rng.gen_range(0..graph_metadata.num_vectors) as u32;
                    (Point::from_f32_vec(unordered_graph_on_storage
                        .read_node(&ramdom_index_1)
                        .unwrap()
                        .0),
                    Point::from_f32_vec(unordered_graph_on_storage
                        .read_node(&ramdom_index_2)
                        .unwrap()
                        .0))
                }).collect()
            }).collect();

            let sum_hit = (0..num_iter).into_iter().map(|_| {
                let ramdom_index = rng.gen_range(0..graph_metadata.num_vectors) as u32;
                let edge_points: Vec<Point> = unordered_graph_on_storage
                    .read_node(&ramdom_index)
                    .unwrap()
                    .1
                    .into_iter()
                    .map(|edge_i| {
                        Point::from_f32_vec(unordered_graph_on_storage
                            .read_node(&edge_i)
                            .unwrap()
                            .0)
                    })
                    .collect();

                let edge_codes: Vec<[Point; 4]> = unordered_graph_on_storage
                    .read_node(&ramdom_index)
                    .unwrap()
                    .1
                    .into_iter()
                    .map(|edge_i| {
                        find_code(&edge_i)
                    })
                    .collect();
                
                let mut edge_pq_dists: Vec<(usize, f32)> = edge_codes.into_iter().map(|edge_code| {
                    edge_code.into_iter().zip(base_pq.iter()).map(|(a, b)| a.distance(b)).sum::<f32>()
                }).enumerate().collect();
                edge_pq_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Less));


                
                let mut dists: Vec<(u8, f32)> = edge_points.clone().into_iter().enumerate().map(|(i, p)| (i as u8, p.distance(&base_point))).collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Less));
                let truth_top_10: Vec<u8> = dists[0..top_k].into_iter().map(|(i, _)| *i).collect();

                let trees: Vec<(Tree, TreeUtils)> = (0..num_tree).map(|iter_count| {
                    let mut tree_utils: TreeUtils = TreeUtils::new(random_splitter_list[iter_count].clone(), 4);
                    let tree = tree_utils.build(edge_points.clone());
                    (tree, tree_utils)
                }).collect();

                let mut resutls: Vec<u8> = trees.into_iter().map(|(tree, tree_utils)| {
                    // let tree_utils: TreeUtils = TreeUtils::new(random_splitter_list[iter_count].clone(), 1);
                    // let tree = tree_utils.build(edge_points.clone());
                    let ann: Vec<u8> = tree_utils.search_tree(tree, &base_point, 0.000).into_iter().map(|(_, i)| i).collect();
                    // println!("{:?}", ann.len());
                    ann
                }).flatten().collect();
                resutls.sort();

                let mut counts = HashMap::new();
                for &number in &resutls {
                    *counts.entry(number).or_insert(0) += 1;
                }
                let counts: Vec<u8> =  counts.into_iter().filter(|(_i, c)| *c >= 13).map(|(i, _)| i).collect();

                // num_tree: 40, depth: 6, ave len: 30, dedup 1
                // 56, 6, 23, 3
                // 27, 5, 22, 3

                // 30, 4, 18, 6

                // 15, 4, 25, 3

                // 70, 4, 15, 13



                // let counts: Vec<u8> = edge_pq_dists[0..30].into_iter().map(|(i, _)| *i as u8).collect();


                let hit = truth_top_10.iter().filter(|i| counts.contains(i)).count();


                println!("hit {}/{}", hit, truth_top_10.len());
                println!("truth_top_k {:?}", truth_top_10);
                println!("{:?}", counts);
                println!("pq_dists len {}\n\n", counts.len());

                if hit == 0 {
                    zero_count += 1
                }

                hit

                // let mut truth_order: Vec<(usize,f32)> = edge_points.iter().map(|p| p.distance(&base_point)).enumerate().collect();
                // truth_order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Less));
                // let truth_order: Vec<usize> = truth_order.into_iter().map(|(i, _)| i).collect();

                // print!("vectune indexing");

                // let (edge_graph, medoid, _backlinks) = vectune::Builder::default()
                //     .set_seed(seed)
                //     .set_a(1.2)
                //     .set_r(5)
                //     .set_l(100)
                //     .build(edge_points);

                // // println!("{:?}", edge_graph.iter().map(|(|_, e)| e).collect::<Vec<&Vec<u32>>>());
                // // println!("medoid: {medoid}, {:?}",edge_graph[medoid as usize].1);
                
                // let ((k_ann, visited), (get_count, waste_count)) = vectune::search_with_analysis_v2(&edge_graph, &base_point, 5, 10, medoid, vec![]);

                // println!("truth : {:?}", &truth_order[0..10]);
                // println!("k-ann : {:?}", &k_ann.into_iter().map(|(_, i)| i).collect::<Vec<u32>>());
                // println!("visited len:{}, waste-rate: {}%", visited.len(), (waste_count as f32/get_count as f32) * 100.0);
            }).sum::<usize>();

            println!("total zero is {}/{}", zero_count, num_iter);
            println!("hit rate {}", sum_hit as f32/ (num_iter * top_k) as f32)
            
        }

        Commands::UnorderRecallRate {
            query_path,
            pq_table_path,
            ground_truth_path,
            unordered_graph_storage_path,
            graph_metadata_path,
            edge_degrees,
            size_l,
        } => {
            let graph_metadata = GraphMetadata::load(&graph_metadata_path).unwrap();
            let pq = PQ::load(&pq_table_path)?;
            let _pq_point_table: [[Point; 256]; 4] = pq
                .cluster_table
                .into_iter()
                .map(|vectors| {
                    vectors
                        .into_iter()
                        .map(|vector| Point::from_f32_vec(vector))
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

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
            );

            let mut graph = UnorderedGraph::new(
                unordered_graph_on_storage,
                graph_metadata.medoid_node_index as u32,
            );

            graph.set_size_l(size_l);

            let query_vector_reader = OriginalVectorReader::new(&query_path)?;
            // let original_vector_reader = OriginalVectorReader::new(&original_vector_path)?;
            let groundtruth: Vec<Vec<u32>> = read_ivecs(&ground_truth_path).unwrap();

            // let query_iter = query_vector_reader.get_num_vectors();
            let query_iter = 100;

            let mut total_time = 0;
            let mut total_waste_count = 0;
            let mut total_get_count = 0;

            let mut _rng = SmallRng::seed_from_u64(rand::random());

            // let mut scores = vec![0 as u32; graph_metadata.num_vectors];

            // let pq_num_divs = 16;
            // let pq_num_divs = 3;


            let hit = (0..query_iter)
                .into_iter()
                .map(|query_index| {
                    // let query_index = rng.gen_range(0..query_vector_reader.get_num_vectors());
                    let query_vector: Vec<f32> = query_vector_reader.read(&query_index).unwrap();
                    println!("query_index {query_index}");

                    // let query_vector: Vec<f32> = vec![rng.gen::<f32>(); 96];

                    let start = Instant::now();
                    // let ((k_ann, visited), (get_count, waste_count)) =
                    //     vectune::search_with_analysis(
                    //         &mut graph,
                    //         &Point::from_f32_vec(query_vector),
                    //         5,
                    //         size_l,
                    //     );
                    // let (get_count, waste_count) = (0, 0);
                    let ((k_ann, visited), (get_count, waste_count)) =
                        vectune::search_with_optimal_stopping(
                            &mut graph,
                            &Point::from_f32_vec(query_vector),
                            5
                        );
                    let t = start.elapsed().as_millis();
                    println!("{t} milli sec, visited len: {}", visited.len());
                    let waste_rate = if waste_count == 0 {
                        0.0
                    } else {
                        (waste_count as f32 / get_count as f32) * 100.0
                    };
                    println!(
                        "waste: {}%",
                        waste_rate
                    );
                    total_time += t;
                    total_waste_count += waste_count;
                    total_get_count += get_count;

                    let result_top_5: Vec<u32> = k_ann.into_iter().map(|(_, i)| i).collect();
                    let top5_groundtruth = &groundtruth[query_index][0..5];
                    println!("{:?}\n{:?}\n\n", top5_groundtruth, result_top_5);
                    let mut hit = 0;
                    for res in result_top_5 {
                        if top5_groundtruth.contains(&res) {
                            hit += 1;
                        }
                    }

                    println!("{hit}/5");

                    // visited
                    //     .into_iter()
                    //     .for_each(|(_, index)| scores[index as usize] += 1);

                    hit
                })
                .reduce(|acc, x| acc + x)
                .unwrap();

            println!("5-recall-rate@5: {}", hit as f32 / (5 * query_iter) as f32);
            println!("average search time: {}", total_time / query_iter as u128);
            println!(
                "total waste: {}%",
                if total_waste_count == 0 {
                    0.0
                } else {
                    (total_waste_count as f32 / total_get_count as f32) * 100.0
                }
            );

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
                        unordered_graph_on_storage.read_node(&(store_index as u32)).unwrap();

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
            let node_byte_size = (vector_dim * 4 + new_edge_degrees * 4 + 4) as usize;
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
                        .write_node(&(index as u32), &vector, &edges)
                        .unwrap();

                    let value = progress_done.fetch_add(1, atomic::Ordering::Relaxed);
                    if value % 1000 == 0 {
                        pb.set_position(value as u64);
                    }
                });

            pb.finish();
        }

        Commands::Pq {
            original_vector_path,
            destination_directory,
            max_chunk_giga_byte_size,
            dataset_size_million,
        } => {
            let _ = max_chunk_giga_byte_size;
            let max_chunk_giga_byte_size = 4;
            let mut original_vector_reader =
                OriginalVectorReader::new_with(&original_vector_path, dataset_size_million)?;

            let (pq, cluster_table): (
                Vec<[k_means::ClusterLabel; 4]>,
                [[k_means::ClusterPoint; k_means::NUM_CLUSTERS]; 4],
            ) = on_disk_k_means_pq(
                &mut original_vector_reader,
                max_chunk_giga_byte_size,
                &mut rng,
            );
            let pq = PQ::new(pq, cluster_table);

            let json = serde_json::to_string(&pq)?;

            let mut file = File::create(format!(
                "{destination_directory}/pq_{dataset_size_million}M.json"
            ))?;
            file.write_all(json.as_bytes())?;
        }

        Commands::TestPq {
            original_vector_path,
            pq_path,
        } => {
            let pq = PQ::load(&pq_path)?;
            let pq_point_table: [[Point; 256]; 4] = pq
                .cluster_table
                .into_iter()
                .map(|vectors| {
                    vectors
                        .into_iter()
                        .map(|vector| Point::from_f32_vec(vector))
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            let pq = pq.pq;

            let original_vector_reader = OriginalVectorReader::new(&original_vector_path)?;

            let to_pq_code = |base_vector: Vec<f32>| -> [u8; 4] {
                let sub_dim = base_vector.len() / 4;
                let base_pq: [u8; 4] = (0..4)
                    .into_iter()
                    .zip(pq_point_table.iter())
                    .map(|(m, clusters)| {
                        let sub_point = Point::from_f32_vec(
                            base_vector[m * sub_dim..(m + 1) * sub_dim].to_vec(),
                        );

                        let c = clusters
                            .iter()
                            .enumerate()
                            .map(|(label, cluster_point)| {
                                (label as u8, cluster_point.distance(&sub_point))
                            })
                            .min_by(|a, b| {
                                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Less)
                            })
                            .unwrap();
                        c.0
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();

                base_pq
            };

            let num_iter = 100;
            let n = 100;
            let sum_hit = (0..num_iter)
                .into_par_iter()
                .map(|_| {
                    let mut rng = thread_rng();
                    let random_index = rng.gen_range(0..pq.len());
                    let base_vector = original_vector_reader.read(&random_index).unwrap();
                    let base_point = Point::from_f32_vec(base_vector.clone());

                    let base_pq = to_pq_code(base_vector);

                    let test_indexies: Vec<usize> = (0..1000).collect();

                    let mut original_order: Vec<(usize, f32)> = test_indexies
                        .iter()
                        .enumerate()
                        .map(|(order_i, vec_index)| {
                            (
                                order_i,
                                Point::from_f32_vec(
                                    original_vector_reader.read(vec_index).unwrap(),
                                )
                                .distance(&base_point),
                            )
                        })
                        .collect();
                    original_order
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Less));

                    let mut pq_order: Vec<(usize, f32)> = test_indexies
                        .iter()
                        .enumerate()
                        .map(|(order_i, vec_index)| {
                            // let test_pq = pq[*vec_index];
                            let test_vector = original_vector_reader.read(vec_index).unwrap();
                            let test_pq = to_pq_code(test_vector);
                            (order_i, dist_pq(&base_pq, &test_pq, &pq_point_table))
                        })
                        .collect();
                    pq_order
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Less));

                    // println!("original_order:   {:?}\npq_order:         {:?}\n\n", original_order, pq_order);

                    let top_n_groundtruth: Vec<usize> =
                        original_order[0..n].iter().map(|(i, _)| *i).collect();
                    let pq_top_n: Vec<usize> = pq_order[0..n].iter().map(|(i, _)| *i).collect();
                    // println!("{:?}\n{:?}\n\n", top_n_groundtruth, pq_top_n);
                    let mut hit = 0;
                    for res in pq_top_n {
                        if top_n_groundtruth.contains(&res) {
                            hit += 1;
                        }
                    }

                    hit
                })
                .sum::<usize>();

            println!(
                "hit: {} %",
                (sum_hit as f32 / (n * num_iter) as f32) * 100.0
            );
        }
    }

    Ok(())
}

fn dist_pq(a: &[u8; 4], b: &[u8; 4], table: &[[Point; 256]; 4]) -> f32 {
    let dist: f32 = a
        .iter()
        .zip(b)
        .zip(table)
        .map(|((a_i, b_i), t)| t[*a_i as usize].distance(&t[*b_i as usize]))
        .sum::<f32>();

    dist
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PQ {
    pq: Vec<[k_means::ClusterLabel; 4]>,
    cluster_table: Vec<Vec<Vec<f32>>>,
    // dist_table: Vec<FxHashMap<(u8, u8), f32>>
}

impl PQ {
    pub fn new(
        pq: Vec<[k_means::ClusterLabel; 4]>,
        cluster_table: [[k_means::ClusterPoint; k_means::NUM_CLUSTERS]; 4],
    ) -> Self {
        Self {
            pq,
            cluster_table: cluster_table
                .into_iter()
                .map(|sub| sub.into_iter().map(|p| p.to_f32_vec()).collect())
                .collect(),
        }
    }

    pub fn load(path: &str) -> Result<Self> {
        let mut file = File::open(path).expect("file not found");

        // ファイルの内容を読み込む
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("something went wrong reading the file");

        // JSONデータをPerson構造体にデシリアライズ
        let pq_table: Self = serde_json::from_str(&contents)?;
        Ok(pq_table)
    }
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

/*

pruneには、個数を制限するロジックはついていない？

todo: external sortのメモリサイズの指定を

cacheは、RefCellを使っているので、thread local出ないといけない

*/
