fn calculate_column_variance_and_mean(data: &Vec<Vec<f32>>) -> Vec<(f32, f32)> {
  let num_columns = data[0].len();

  let mut columns = vec![Vec::new(); num_columns];

  // Organize data by columns
  for row in data {
      for (i, &value) in row.iter().enumerate() {
          columns[i].push(value);
      }
  }

  // Calculate variances and medians
  let mut results = Vec::new();
  for column in columns {
      let mean = column.iter().sum::<f32>() / column.len() as f32;
      let variance = column
          .iter()
          .map(|&value| {
              let diff = value - mean;
              diff * diff
          })
          .sum::<f32>()
          / column.len() as f32;

      // Sort the column to find the median
      let mut sorted_column = column.clone();
      sorted_column.sort_by(|a, b| a.partial_cmp(b).unwrap());

      let median = if sorted_column.len() % 2 == 0 {
          let mid = sorted_column.len() / 2;
          (sorted_column[mid - 1] + sorted_column[mid]) / 2.0
      } else {
          sorted_column[sorted_column.len() / 2]
      };

      results.push((variance, median));
  }

  results
}

fn split_data_by_median(
  data: Vec<Vec<f32>>,
  median: f32,
  index: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
  let mut below = Vec::new();
  let mut above = Vec::new();

  for row in data {
      if row[index] <= median {
          below.push(row);
      } else {
          above.push(row);
      }
  }

  (below, above)
}

enum Tree {
  Node(usize, f32, Box<Tree>, Box<Tree>),
  Leaf(Vec<Vec<f32>>),
}

fn build_kd_tree(vectors: Vec<Vec<f32>>, depth: usize) -> Tree {
  let (max_variance_index, &(_, median)) =
      calculate_column_variance_and_mean(&vectors)
          .iter()
          .enumerate()
          .max_by(|(_, &(var1, _)), (_, &(var2, _))| var1.partial_cmp(&var2).unwrap())
          .unwrap();

  let (below, above) = split_data_by_median(vectors, median, max_variance_index);

  if depth == 1 {
      Tree::Node(
          max_variance_index,
          median,
          Box::new(Tree::Leaf(below)),
          Box::new(Tree::Leaf(above)),
      )
  } else {
      Tree::Node(
          max_variance_index,
          median,
          Box::new(build_kd_tree(below, depth - 1)),
          Box::new(build_kd_tree(above, depth - 1)),
      )
  }
}

fn search_kd_tree(kd_tree: Tree, query: &Vec<f32>) -> Vec<Vec<f32>> {
  match kd_tree {
      Tree::Node(v_index, median, below, above) => {
          if query[v_index] <= median {
              search_kd_tree(*below, query)
          } else {
              search_kd_tree(*above, query)
          }
      },
      Tree::Leaf(vectors) => vectors
  }
}


        // // depth 3
        // let Tree::Node(s7, n14, n15)    = *n6   else {panic!()};
        // let Tree::Node(s8, n16, n17)    = *n7   else {panic!()};
        // let Tree::Node(s9, n18, n19)    = *n8   else {panic!()};
        // let Tree::Node(s10,n20, n21)    = *n9   else {panic!()};
        // let Tree::Node(s11,n22, n23)    = *n10  else {panic!()};
        // let Tree::Node(s12,n24, n25)    = *n11  else {panic!()};
        // let Tree::Node(s13,n26, n27)    = *n12  else {panic!()};
        // let Tree::Node(s14,n28, n29)    = *n13  else {panic!()};


// let ramdom_base_index = rng.gen_range(0..graph_metadata.num_vectors);
// let base_vector = unordered_graph_on_storage
// .read_node(&ramdom_base_index)
// .unwrap()
// .0;
// let base_point = Point::from_f32_vec(
//     base_vector.clone()
// );

// let num_iter = 100;

// (0..num_iter).into_iter().for_each(|_| {
//     let ramdom_index = rng.gen_range(0..graph_metadata.num_vectors);
//     let edge_vectors: Vec<Vec<f32>> = unordered_graph_on_storage
//         .read_node(&ramdom_index)
//         .unwrap()
//         .1
//         .into_iter()
//         .map(|edge_i| {
//             unordered_graph_on_storage
//                 .read_node(&(edge_i as usize))
//                 .unwrap()
//                 .0
//         })
//         .collect();
// });
