use crate::graph_store::GraphStore;
use crate::original_vector_reader::OriginalVectorReaderTrait;
use crate::point::Point;
use crate::storage::Storage;
use indicatif::ProgressBar;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

pub fn single_index<R: OriginalVectorReaderTrait + std::marker::Sync>(
    vector_reader: &R,
    graph_on_storage: &GraphStore<Storage>,
    seed: u64,
) {
    let shard_points: Vec<Point> = (0..vector_reader.get_num_vectors())
        .into_par_iter()
        .map(|node_id| Point::from_f32_vec(vector_reader.read(&node_id).unwrap()))
        .collect();
    // 2. vectune::indexに渡すノードのindexとidとのtableを作る
    let (indexed_shard, _start_shard_id, _): (Vec<(Point, Vec<u32>)>, u32, Vec<Vec<u32>>) =
        vectune::Builder::default()
            .set_seed(seed)
            .set_a(3.0)
            .progress(ProgressBar::new(1000))
            .build(shard_points);
    // let _start_node_id = table_for_shard_id_to_node_id[start_shard_id as usize];

    // 3. idをもとにssdに書き込む。
    // 4. bitmapを持っておいて、idがtrueの時には、すでにあるedgesをdeserializeして、extend, dup。
    indexed_shard
        .into_par_iter()
        .enumerate()
        .for_each(|(node_id, (point, edges))| {
            graph_on_storage
                .write_node(&node_id, &point.to_f32_vec(), &edges)
                .unwrap();
        });
}

// WIP:: test sharded_index
#[cfg(test)]
mod test {
    // #[test]
    // fn testing_sharded_index() {}
}
