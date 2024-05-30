// #![feature(portable_simd)]

#[cfg(test)]
mod tests;

pub mod graph_store;
pub mod k_means;
pub mod original_vector_reader;
pub mod point;
pub mod sharded_index;
pub mod storage;
pub mod utils;

use std::path;

use anyhow::Result;
use bit_vec::BitVec;
use bytesize::GB;
use ext_sort::{buffer::LimitedBufferBuilder, ExternalSorter, ExternalSorterBuilder};
use itertools::Itertools;
use k_means::on_disk_k_means;
// use node_reader::{EdgesIterator, GraphOnStorage, GraphOnStorageTrait};
use original_vector_reader::{OriginalVectorReader, OriginalVectorReaderTrait};
use rand::{rngs::SmallRng, SeedableRng};
use sharded_index::sharded_index;

use crate::{
    graph_store::{EdgesIterator, GraphStore},
    storage::Storage,
};

type VectorIndex = usize;

fn main() -> Result<()> {
    let seed = 01234;
    let mut rng = SmallRng::seed_from_u64(seed);

    /* k-meansをSSD上で行う */
    println!("reading vector file");
    let path = "test_vectors/base.10M.fbin";
    // 10Mだと大きすぎるので、小さなデータセットをここから作る。
    let vector_reader = OriginalVectorReader::new(path)?;

    println!("k-menas on disk");
    let num_clusters: u8 = 16;
    let cluster_labels = on_disk_k_means(&vector_reader, &num_clusters, &mut rng);

    /* sharding */
    println!("sharded indexing");
    // let mut graph_on_stroage = GraphStore::new_with_empty_file(
    //     "test_vectors/unordered_graph.10M.graph",
    //     vector_reader.get_num_vectors() as u32,
    //     vector_reader.get_vector_dim() as u32,
    //     70 * 2,
    // )?;

    let node_byte_size = (96 * 4 + 140 * 4 + 4) as usize;
    let file_byte_size = 11 * 1000000 * node_byte_size;
    let num_node_in_sector = 10;
    let sector_byte_size = num_node_in_sector * node_byte_size;
    let storage = Storage::new_with_empty_file(
        "test_vectors/unordered_graph.10M.graph",
        file_byte_size as u64,
        sector_byte_size,
    )
    .unwrap();
    let mut graph_on_stroage = GraphStore::new(
        vector_reader.get_num_vectors(),
        vector_reader.get_vector_dim(),
        70 * 2,
        storage,
    );

    sharded_index(
        &vector_reader,
        &mut graph_on_stroage,
        &num_clusters,
        &cluster_labels,
        seed,
    );

    /* gordering */
    // 2. 並び替えの順番をもとに、ssdに書き込む。
    println!("gordering");
    let edge_iter = EdgesIterator::new(&graph_on_stroage);
    let sorter: ExternalSorter<(u32, u32), std::io::Error, LimitedBufferBuilder> =
        ExternalSorterBuilder::new()
            .with_tmp_dir(path::Path::new("./"))
            // .with_buffer(LimitedBufferBuilder::new((50 * MB) as usize, false))
            .build()
            .unwrap();
    let sorted = sorter.sort_by(edge_iter, |a, b| a.0.cmp(&b.0)).unwrap();
    let backlinks: Vec<Vec<u32>> = sorted
        .into_iter()
        .map(Result::unwrap)
        .chunk_by(|&(key, _)| key)
        .into_iter()
        .map(|(_key, group)| group.map(|(_, edge)| edge).collect())
        .collect();

    let get_backlinks = |id: &u32| -> Vec<u32> { backlinks[*id as usize].clone() };

    let get_edges =
        |id: &u32| -> Vec<u32> { graph_on_stroage.read_edges(&(*id as usize)).unwrap() };

    let target_node_bit_vec = BitVec::from_elem(vector_reader.get_num_vectors(), true);
    let window_size = num_node_in_sector;
    let reordered_node_ids = vectune::gorder(
        get_edges,
        get_backlinks,
        target_node_bit_vec,
        window_size,
        &mut rng,
    );

    /* diskへの書き込み */
    let file_byte_size = 10 * GB; // wip
    let storage = Storage::new_with_empty_file(
        "./test_vectors/ordered_graph.10M.graph",
        file_byte_size,
        sector_byte_size,
    )
    .unwrap();
    let ordered_graph_on_storage = GraphStore::new(
        vector_reader.get_num_vectors(),
        vector_reader.get_vector_dim(),
        70 * 2,
        storage,
    );

    reordered_node_ids
        .chunks(window_size)
        .enumerate()
        .for_each(|(sector_index, node_ids)| {
            let mut node_offset = 0;
            let mut buffer: Vec<u8> = vec![0; sector_byte_size];
            for node_id in node_ids {
                let serialized_node = graph_on_stroage.read_serialized_node(&(*node_id as usize));
                let node_offset_end = node_byte_size;
                buffer[node_offset..node_offset_end].copy_from_slice(&serialized_node);
                node_offset = node_offset_end;
            }

            ordered_graph_on_storage
                .write_serialized_sector(&sector_index, &buffer)
                .unwrap();
        });

    /* node_id とstore_indexの変換表を作る。 */

    Ok(())
}

/*
Note:
storageに書き込まれているedgesは、確保されている要素数に満たない場合があり、それはどうdeserializeされるのか。

*/
