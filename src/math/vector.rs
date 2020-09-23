use crate::math::num::RealNumber;
use std::collections::HashMap;

pub trait RealNumberVector<T: RealNumber> {
    fn unique(&self) -> (Vec<T>, Vec<usize>);
}

impl<T: RealNumber> RealNumberVector<T> for Vec<T> {
    fn unique(&self) -> (Vec<T>, Vec<usize>) {
        let mut unique = self.clone();
        unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique.dedup();

        let mut index = HashMap::with_capacity(unique.len());
        for (i, u) in unique.iter().enumerate() {
            index.insert(u.to_i64().unwrap(), i);
        }

        let mut unique_index = Vec::with_capacity(self.len());
        for e in self {
            unique_index.push(index[&e.to_i64().unwrap()]);
        }

        (unique, unique_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unique() {
        let v1 = vec![0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 4.0];
        assert_eq!(
            (vec!(0.0, 1.0, 2.0, 4.0), vec!(0, 0, 1, 1, 2, 0, 3)),
            v1.unique()
        );
    }
}
