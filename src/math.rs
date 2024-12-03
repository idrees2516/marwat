use crate::types::{OptionType, Pool, Strike};
use ethers::types::U256;
use num_traits::Zero;
use std::cmp;

const Q96: U256 = U256([0x1000000000000000000, 0, 0, 0]); // 2^96
const Q192: U256 = U256([0, 0x1000000000000000000, 0, 0]); // 2^192

pub fn calculate_premium(
    pool: &Pool,
    option_type: OptionType,
    strike: Strike,
    amount: U256,
) -> U256 {
    let spot_price = pool.sqrt_price_x96.pow(2.into()) / Q96;
    let time_value = calculate_time_value(pool, strike);
    let intrinsic_value = match option_type {
        OptionType::Call => {
            if spot_price > strike.0 {
                spot_price - strike.0
            } else {
                U256::zero()
            }
        }
        OptionType::Put => {
            if strike.0 > spot_price {
                strike.0 - spot_price
            } else {
                U256::zero()
            }
        }
    };

    (intrinsic_value + time_value) * amount
}

pub fn calculate_collateral(
    pool: &Pool,
    option_type: OptionType,
    strike: Strike,
    amount: U256,
) -> U256 {
    let spot_price = pool.sqrt_price_x96.pow(2.into()) / Q96;
    
    match option_type {
        OptionType::Call => {
            // For calls, collateral is max(spot_price - strike, 0) * amount
            if spot_price > strike.0 {
                (spot_price - strike.0) * amount
            } else {
                U256::zero()
            }
        }
        OptionType::Put => {
            // For puts, collateral is strike * amount
            strike.0 * amount
        }
    }
}

fn calculate_time_value(pool: &Pool, strike: Strike) -> U256 {
    let spot_price = pool.sqrt_price_x96.pow(2.into()) / Q96;
    let volatility = calculate_volatility(pool);
    
    // Simplified time value calculation based on Black-Scholes approximation
    let strike_distance = if strike.0 > spot_price {
        strike.0 - spot_price
    } else {
        spot_price - strike.0
    };

    let base_time_value = (volatility * spot_price) / Q96;
    let time_decay = Q96 - (strike_distance * Q96 / spot_price);
    
    (base_time_value * time_decay) / Q96
}

fn calculate_volatility(pool: &Pool) -> U256 {
    // Calculate implied volatility from pool liquidity and tick spacing
    let liquidity_factor = (pool.liquidity * Q96) / pool.sqrt_price_x96;
    let tick_factor = U256::from(pool.tick_spacing.abs());
    
    // Simplified volatility calculation
    (liquidity_factor * tick_factor) / Q96
}

pub fn get_next_sqrt_price_from_input(
    sqrt_price_x96: U256,
    liquidity: U256,
    amount_in: U256,
    zero_for_one: bool,
) -> U256 {
    if zero_for_one {
        let numerator = liquidity * Q96 * sqrt_price_x96;
        let denominator = liquidity * Q96 + amount_in * sqrt_price_x96;
        numerator / denominator
    } else {
        let product = sqrt_price_x96 * amount_in;
        let sum = liquidity * Q96 + product;
        let quotient = (product * sqrt_price_x96) / sum;
        sqrt_price_x96 + quotient
    }
}

pub fn get_next_sqrt_price_from_output(
    sqrt_price_x96: U256,
    liquidity: U256,
    amount_out: U256,
    zero_for_one: bool,
) -> U256 {
    if zero_for_one {
        let sum = liquidity * Q96 + amount_out * sqrt_price_x96;
        let product = sqrt_price_x96 * amount_out;
        sqrt_price_x96 - (product / sum)
    } else {
        let numerator = liquidity * Q96 * sqrt_price_x96;
        let denominator = liquidity * Q96 - amount_out * sqrt_price_x96;
        numerator / denominator
    }
}

pub fn get_amount_out(
    sqrt_price_x96: U256,
    sqrt_price_target_x96: U256,
    liquidity: U256,
    zero_for_one: bool,
) -> U256 {
    if zero_for_one {
        let diff = sqrt_price_x96 - sqrt_price_target_x96;
        (liquidity * diff) / Q96
    } else {
        let diff = sqrt_price_target_x96 - sqrt_price_x96;
        (liquidity * diff) / Q96
    }
}

pub fn get_amount_in(
    sqrt_price_x96: U256,
    sqrt_price_target_x96: U256,
    liquidity: U256,
    zero_for_one: bool,
) -> U256 {
    if zero_for_one {
        let numerator = liquidity * Q96 * (sqrt_price_x96 - sqrt_price_target_x96);
        let denominator = sqrt_price_target_x96 * sqrt_price_x96;
        numerator / denominator
    } else {
        let numerator = liquidity * (sqrt_price_target_x96 - sqrt_price_x96);
        numerator / Q96
    }
}
