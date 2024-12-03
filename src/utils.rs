use ethers::types::{Address, U256};
use sha3::{Digest, Keccak256};

pub fn sort_tokens(token_a: Address, token_b: Address) -> (Address, Address) {
    if token_a < token_b {
        (token_a, token_b)
    } else {
        (token_b, token_a)
    }
}

pub fn compute_pool_address(
    factory: Address,
    token0: Address,
    token1: Address,
    fee: u32,
) -> Address {
    let mut hasher = Keccak256::new();
    
    // Pack the data according to the CREATE2 specification
    hasher.update(b"ff");
    hasher.update(factory.as_bytes());
    hasher.update(&encode_pool_init_code_hash(token0, token1, fee));
    
    let hash = hasher.finalize();
    Address::from_slice(&hash[12..])
}

fn encode_pool_init_code_hash(
    token0: Address,
    token1: Address,
    fee: u32,
) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    
    // Pack the pool initialization parameters
    hasher.update(token0.as_bytes());
    hasher.update(token1.as_bytes());
    hasher.update(&fee.to_be_bytes());
    
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&hasher.finalize());
    hash
}

pub fn encode_price_sqrt_x96(price: f64) -> U256 {
    let price_sqrt = price.sqrt();
    U256::from_f64_lossy(price_sqrt * 2f64.powi(96))
}

pub fn decode_price_sqrt_x96(sqrt_price_x96: U256) -> f64 {
    let price_sqrt = sqrt_price_x96.as_u128() as f64 / 2f64.powi(96);
    price_sqrt * price_sqrt
}

pub fn tick_to_price(tick: i32) -> f64 {
    1.0001f64.powi(tick)
}

pub fn price_to_tick(price: f64) -> i32 {
    (price.ln() / 0.0001f64.ln()).round() as i32
}

pub fn get_min_tick(tick_spacing: i32) -> i32 {
    -887272 / tick_spacing * tick_spacing
}

pub fn get_max_tick(tick_spacing: i32) -> i32 {
    887272 / tick_spacing * tick_spacing
}

pub fn is_valid_tick(tick: i32, tick_spacing: i32) -> bool {
    tick % tick_spacing == 0 && tick >= get_min_tick(tick_spacing) && tick <= get_max_tick(tick_spacing)
}

pub fn compute_swap_step(
    sqrt_price_x96: U256,
    sqrt_price_target_x96: U256,
    liquidity: U256,
    amount: U256,
    zero_for_one: bool,
) -> (U256, U256, U256) {
    use crate::math::{get_amount_in, get_amount_out, get_next_sqrt_price_from_input, get_next_sqrt_price_from_output};

    let sqrt_price_next_x96 = if zero_for_one {
        get_next_sqrt_price_from_input(sqrt_price_x96, liquidity, amount, true)
    } else {
        get_next_sqrt_price_from_output(sqrt_price_x96, liquidity, amount, false)
    };

    let amount_in = get_amount_in(sqrt_price_x96, sqrt_price_next_x96, liquidity, zero_for_one);
    let amount_out = get_amount_out(sqrt_price_x96, sqrt_price_next_x96, liquidity, zero_for_one);

    (sqrt_price_next_x96, amount_in, amount_out)
}

pub fn compute_premium_and_collateral(
    option_type: crate::types::OptionType,
    strike: crate::types::Strike,
    amount: U256,
    pool: &crate::types::Pool,
) -> (U256, U256) {
    use crate::math::{calculate_premium, calculate_collateral};

    let premium = calculate_premium(pool, option_type, strike, amount);
    let collateral = calculate_collateral(pool, option_type, strike, amount);

    (premium, collateral)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_tokens() {
        let token_a = Address::from_low_u64_be(2);
        let token_b = Address::from_low_u64_be(1);
        let (token0, token1) = sort_tokens(token_a, token_b);
        assert!(token0 < token1);
    }

    #[test]
    fn test_tick_to_price() {
        let tick = 1;
        let price = tick_to_price(tick);
        assert!((price - 1.0001).abs() < 1e-10);
    }

    #[test]
    fn test_price_to_tick() {
        let price = 1.0001;
        let tick = price_to_tick(price);
        assert_eq!(tick, 1);
    }

    #[test]
    fn test_tick_spacing_bounds() {
        let tick_spacing = 1;
        assert_eq!(get_min_tick(tick_spacing), -887272);
        assert_eq!(get_max_tick(tick_spacing), 887272);
    }

    #[test]
    fn test_valid_tick() {
        let tick_spacing = 60;
        assert!(is_valid_tick(60, tick_spacing));
        assert!(!is_valid_tick(61, tick_spacing));
    }
}
