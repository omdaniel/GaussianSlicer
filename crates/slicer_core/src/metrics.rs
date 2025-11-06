//! Numeric validation helpers for comparing CPU/GPU outputs (placeholder).

use crate::Scalar;

/// Computes a simple checksum over density samples to aid parity testing.
pub fn checksum(values: &[Scalar]) -> Scalar {
    values.iter().copied().fold(0.0, |acc, v| acc + v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_is_linear() {
        assert_eq!(checksum(&[1.0, 2.0, 3.0]), 6.0);
    }
}
