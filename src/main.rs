#![cfg_attr(feature = "simd", feature(portable_simd))]

#[cfg(test)]
mod tests;

pub mod graph;
pub mod graph_store;
pub mod k_means;
pub mod original_vector_reader;
pub mod point;
pub mod sharded_index;
pub mod single_index;
pub mod storage;
pub mod utils;

use std::path;

use anyhow::Result;
use bit_vec::BitVec;
use bytesize::GB;
use ext_sort::{buffer::LimitedBufferBuilder, ExternalSorter, ExternalSorterBuilder};
use graph::Graph;
use itertools::Itertools;
use rayon::{
    iter::{IntoParallelIterator, ParallelBridge, ParallelIterator},
    vec,
};

use k_means::on_disk_k_means;
// use node_reader::{EdgesIterator, GraphOnStorage, GraphOnStorageTrait};
use original_vector_reader::{OriginalVectorReader, OriginalVectorReaderTrait};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use sharded_index::sharded_index;
use single_index::single_index;
use vectune::{GraphInterface, PointInterface};

use crate::{
    graph_store::{EdgesIterator, GraphStore},
    point::Point,
    storage::Storage,
};

type VectorIndex = usize;

fn main() -> Result<()> {
    let seed = 01234;
    let mut rng = SmallRng::seed_from_u64(seed);

    let do_indexing = true;

    /* read original vectors */
    println!("reading vector file");
    let path = "test_vectors/base.10M.fbin";
    // 10Mだと大きすぎるので、小さなデータセットをここから作る。
    let vector_reader = OriginalVectorReader::new_with_1m(path)?;

    /* sharding */
    println!("initializing graph");
    let node_byte_size = (96 * 4 + 140 * 4 + 4) as usize;
    let file_byte_size = 11 * 1000000 * node_byte_size;
    let num_node_in_sector = 10;
    let sector_byte_size = num_node_in_sector * node_byte_size;

    let storage = if do_indexing {
        Storage::new_with_empty_file(
            "test_vectors/unordered_graph.10M.graph",
            file_byte_size as u64,
            sector_byte_size,
        )
        .unwrap()
    } else {
        Storage::load("test_vectors/unordered_graph.10M.graph", sector_byte_size).unwrap()
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
    let (cluster_labels, cluster_points) = on_disk_k_means(&vector_reader, &num_clusters, &mut rng);
    if do_indexing {
        sharded_index(
            &vector_reader,
            &mut graph_on_stroage,
            &num_clusters,
            &cluster_labels,
            seed,
        );

        // single_index(
        //     &vector_reader,
        //     &mut graph_on_stroage,
        //     seed,
        // );

        /*

        cluster pointの平均を全部の平均点として、それをstart_pointとする。
        これの検索は、最も近いclusterのnode_idを開始点として探索して、そこで見つかったnodeを全体のスタートとする

        */
    } else {
        println!("skiped");
    }

    // for store_id in 0..vector_reader.get_num_vectors() {
    //     let edges = graph_on_stroage.read_edges(&store_id).unwrap();
    //     if edges.len() == 0 {
    //         print!("edges len is 0, index: {}", store_id)
    //     }
    // }

    // println!("edges: {:?}", graph_on_stroage.read_node(&190));

    /* gordering */
    // 2. 並び替えの順番をもとに、ssdに書き込む。

    // WIP: buffer size
    let buffer_size = 2 * GB as usize;

    println!("gordering");
    let edge_iter = EdgesIterator::new(&graph_on_stroage);

    println!("make sorter");
    let sorter: ExternalSorter<(u32, u32), std::io::Error, LimitedBufferBuilder> =
        ExternalSorterBuilder::new()
            .with_tmp_dir(path::Path::new("./"))
            .with_buffer(LimitedBufferBuilder::new(buffer_size, false))
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
        let file_byte_size = 10 * GB; // wip
        println!("Storage::new_with_empty_file");
        Storage::new_with_empty_file(
            "./test_vectors/ordered_graph.10M.graph",
            file_byte_size,
            sector_byte_size,
        )
        .unwrap()
    } else {
        println!("skiped");
        Storage::load("./test_vectors/ordered_graph.10M.graph", sector_byte_size).unwrap()
    };

    println!("GraphStore::new");
    let ordered_graph_on_storage = GraphStore::new(
        vector_reader.get_num_vectors(),
        vector_reader.get_vector_dim(),
        70 * 2,
        storage,
    );

    println!("reordered_node_ids");
    reordered_node_ids
        .chunks(window_size)
        .enumerate()
        .par_bridge()
        .for_each(|(sector_index, node_ids)| {
            let mut node_offset = 0;
            let mut buffer: Vec<u8> = vec![0; sector_byte_size];
            for node_id in node_ids {
                let serialized_node = graph_on_stroage.read_serialized_node(&(*node_id as usize));
                let node_offset_end = node_offset + node_byte_size;
                buffer[node_offset..node_offset_end].copy_from_slice(&serialized_node);
                node_offset = node_offset_end;
            }

            ordered_graph_on_storage
                .write_serialized_sector(&sector_index, &buffer)
                .unwrap();
        });

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

    // let (dist, index) = (0..vector_reader.get_num_vectors())
    //     .into_par_iter()
    //     .map(|index| {
    //         let (vector, _) = graph_on_stroage.read_node(&index).unwrap();
    //         (ave_point.distance(&Point::from_f32_vec(vector)), index)
    //     })
    //     .reduce_with(|acc, x| if acc.0 < x.0 { acc } else { x })
    //     .unwrap();
    // println!("全て計算している方:centroid_node_index: {} {}", index, dist);

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
        node_index_to_store_index,
        tmp_start_index as u32,
    );

    let (dist, centroid_node_index) = vectune::search(&mut graph, &ave_point, 10).0[0];

    println!(
        "centroid_node_index: {}. dist: {}",
        centroid_node_index, dist
    );

    graph.set_start_node_index(centroid_node_index);

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

    Ok(())
}

/*
Note:
storageに書き込まれているedgesは、確保されている要素数に満たない場合があり、それはどうdeserializeされるのか。

pruneには、個数を制限するロジックはついていない？

todo: external sortのメモリサイズの指定を

*/
