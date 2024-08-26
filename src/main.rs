#![cfg_attr(feature = "simd-l1", feature(portable_simd))]

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
pub mod embed;

use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use anyhow::Result;
use embed::EmbeddingModel;
use graph::Graph;
// use graph_store::GraphHeader;
use original_vector_reader::{read_ivecs, OriginalVectorReader, OriginalVectorReaderTrait};
use single_index::single_index;
use vectune::PointInterface;

use crate::{graph_store::GraphStore, point::Point, storage::Storage};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    SingleShardBuild {
        #[arg(long)]
        fbin: String,
        #[arg(long)]
        dir: String,
        #[arg(long)]
        max_edge_degrees: usize,
        #[arg(long)]
        seed: u64,
    },

    RecallRate {
        #[arg(long)]
        graph_storage_path: String,
        #[arg(long)]
        ground_truth_ivecs_path: String,
        #[arg(long)]
        query_fbin_path: String,
        // #[arg(long)]
        // seed: u64,
        #[arg(long)]
        query_iter: usize,
    },

    EmbedSentences {
        #[arg(long)]
        sentences_path: String,
        #[arg(long)]
        out: String,
        #[arg(long)]
        batch_size: usize,
    },
}

// use serde::{Deserialize, Serialize};

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        // cargo run --release  --features debug -- single-shard-build --fbin /Volumes/WD_BLACK/index_deep1b/base.1B.fbin --dir test_vectors/2024_8_23 --max-edge-degrees 70 --seed 12345
        Commands::SingleShardBuild {
            fbin,
            dir,
            // dataset_size,
            max_edge_degrees,
            seed,
        } => {
            /* file path */
            let dir = Path::new(&dir);
            let graph_storage_path =
                // format!("{destination_directory}/graph.bin");
                dir.join("graph.bin");
            let backlinks_path = dir.join("backlinks.bin");

            /* read original vectors */
            println!("reading vector file");
            #[cfg(feature = "debug")]
            let vector_reader = {
                println!("dataset_size: {} million", 1);
                OriginalVectorReader::new_with(&fbin, 1)?
            };
            #[cfg(not(feature = "debug"))]
            let vector_reader = OriginalVectorReader::new(&fbin)?;

            /* index */
            println!("initializing graph");
            let node_byte_size =
                utils::node_byte_size(vector_reader.get_vector_dim(), max_edge_degrees);
            let file_byte_size = utils::file_byte_size(node_byte_size, vector_reader.get_num_vectors());

            let storage =
                Storage::new_with_empty_file(&graph_storage_path, file_byte_size.clone() as u64)
                    .unwrap();

            let mut graph_on_storage = GraphStore::new(
                vector_reader.get_num_vectors(),
                vector_reader.get_vector_dim(),
                max_edge_degrees,
                storage,
            );

            /* build and write backlinks */
            let (_start_id, backlinks) = single_index(&vector_reader, &mut graph_on_storage, seed);
            let backlinks_bytes = bincode::serialize(&backlinks)?;
            let mut file = File::create(backlinks_path)?;
            file.write_all(&backlinks_bytes)?;
        }


        //  cargo run --release  --features debug -- recall-rate --graph-storage-path test_vectors/2024_8_23/graph.bin --ground-truth-ivecs-path test_vectors/gt/deep1M_groundtruth.ivecs --query-fbin-path test_vectors/query.public.10K.fbin --query-iter 10
        Commands::RecallRate {
            graph_storage_path,
            ground_truth_ivecs_path,
            query_fbin_path,
            query_iter,
            // seed,
        } => {
            /* recall-rate */
            let mut graph = Graph::new(GraphStore::load(
                Storage::load(&graph_storage_path).unwrap(),
            ));

            let query_vector_reader =
                OriginalVectorReader::new(&query_fbin_path)?;
            let groundtruth: Vec<Vec<u32>> =
                read_ivecs(&ground_truth_ivecs_path).unwrap();

            let k = 5;

            let hit = (0..query_iter)
                .into_iter()
                .map(|query_index| {
                    let query_vector: Vec<f32> = query_vector_reader.read(&query_index).unwrap();
                    let k_ann =
                        vectune::search(&mut graph, &Point::from_f32_vec(query_vector), k).0;

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

            println!("5-recall-rate@5: {}", hit as f32 / (k * query_iter) as f32);
        }

        // ❯ cargo run --release  -- embed-sentences --sentences-path /Users/clankpan/Develop/Kinic/auto_repository_retriever/debug/3505716682c93b8662fe472c9524243e607b0611_chunk_text_vec.json --batch-size 10 --out test_vectors/topic.fbin
        Commands::EmbedSentences { sentences_path, out , batch_size} => {

            let mut model = EmbeddingModel::new()?;

            let sentences: Vec<String> = serde_json::from_str(&open_file_as_string(Path::new(&sentences_path))?)?;
            let embeddings: Vec<Vec<f32>> = sentences.chunks(batch_size).enumerate().map(|(index, sentences)| {
                println!("{index}");
                model.get_embeddings(&sentences.to_vec(), true).expect("msg")
            }).flatten().collect();

            let num_vectors = embeddings.len();
            let vector_dim = 384;

            let num_vectors_bytes = (num_vectors as u32).to_le_bytes();
            let vector_dim_bytes = (vector_dim as u32).to_le_bytes();

            // header
            let mut bytes = Vec::with_capacity(8 + num_vectors * vector_dim * 4);
            bytes.extend_from_slice(&num_vectors_bytes);
            bytes.extend_from_slice(&vector_dim_bytes);

            // body
            let flat_data: Vec<f32> = embeddings.into_iter().flatten().collect();
            bytes.extend_from_slice(bytemuck::cast_slice(&flat_data));

            File::create(out)?.write_all(&bytes)?;
        }
    }

    Ok(())
}

fn open_file_as_string(path: &Path) -> Result<String> {
    let mut content = String::new();
    File::open(path)?.read_to_string(&mut content)?;
    Ok(content)
}