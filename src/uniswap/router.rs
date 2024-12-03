use ethers::types::{U256, Address, H256};
use std::collections::HashMap;
use std::sync::Arc;
use crate::types::{Result, PanopticError};
use super::pool::{UniswapV3Pool, SwapParams};

/// Router for managing swaps across multiple Uniswap V3 pools
pub struct UniswapV3Router {
    pools: HashMap<Address, Arc<UniswapV3Pool>>,
    fee_tiers: Vec<u32>,
}

/// Parameters for finding the optimal swap path
#[derive(Debug)]
pub struct PathParams {
    pub token_in: Address,
    pub token_out: Address,
    pub amount_in: U256,
    pub max_hops: usize,
}

/// Represents a swap path
#[derive(Debug, Clone)]
pub struct SwapPath {
    pub pools: Vec<Address>,
    pub tokens: Vec<Address>,
    pub fees: Vec<u32>,
    pub amounts: Vec<U256>,
}

impl UniswapV3Router {
    pub fn new(fee_tiers: Vec<u32>) -> Self {
        Self {
            pools: HashMap::new(),
            fee_tiers,
        }
    }

    /// Adds a pool to the router
    pub fn add_pool(&mut self, pool: UniswapV3Pool) {
        self.pools.insert(pool.address, Arc::new(pool));
    }

    /// Finds the optimal path for a swap
    pub fn find_optimal_path(&self, params: PathParams) -> Result<SwapPath> {
        let mut best_path = None;
        let mut best_output = U256::zero();

        // Try all possible paths up to max_hops
        let paths = self.enumerate_paths(params.token_in, params.token_out, params.max_hops)?;

        for path in paths {
            let (amounts, total_output) = self.simulate_path_swap(&path, params.amount_in)?;
            
            if total_output > best_output {
                best_output = total_output;
                best_path = Some(SwapPath {
                    pools: path.pools,
                    tokens: path.tokens,
                    fees: path.fees,
                    amounts,
                });
            }
        }

        best_path.ok_or_else(|| PanopticError::NoPathFound)
    }

    /// Simulates a swap along a path
    fn simulate_path_swap(
        &self,
        path: &SwapPath,
        amount_in: U256,
    ) -> Result<(Vec<U256>, U256)> {
        let mut amounts = vec![amount_in];
        let mut current_amount = amount_in;

        for (i, &pool_address) in path.pools.iter().enumerate() {
            let pool = self.pools.get(&pool_address)
                .ok_or(PanopticError::PoolNotFound)?;

            let zero_for_one = pool.token0 == path.tokens[i];
            
            let params = SwapParams {
                zero_for_one,
                amount_specified: current_amount.as_u128() as i128,
                sqrt_price_limit_x96: if zero_for_one {
                    U256::from(4295128739) // MIN_SQRT_RATIO + 1
                } else {
                    U256::from(1461446703485210103287273052203988822378723970342) // MAX_SQRT_RATIO - 1
                },
            };

            let (amount0, amount1, _, _) = pool.simulate_swap(params)?;
            
            current_amount = U256::from((if zero_for_one { amount1 } else { amount0 }).unsigned_abs());
            amounts.push(current_amount);
        }

        Ok((amounts, *amounts.last().unwrap()))
    }

    /// Enumerates all possible paths between two tokens
    fn enumerate_paths(
        &self,
        token_in: Address,
        token_out: Address,
        max_hops: usize,
    ) -> Result<Vec<SwapPath>> {
        let mut paths = Vec::new();
        let mut current_path = SwapPath {
            pools: Vec::new(),
            tokens: vec![token_in],
            fees: Vec::new(),
            amounts: Vec::new(),
        };

        self.find_paths(
            token_in,
            token_out,
            max_hops,
            &mut current_path,
            &mut paths,
            &mut std::collections::HashSet::new(),
        )?;

        Ok(paths)
    }

    /// Recursive helper for path enumeration
    fn find_paths(
        &self,
        current_token: Address,
        token_out: Address,
        hops_remaining: usize,
        current_path: &mut SwapPath,
        paths: &mut Vec<SwapPath>,
        visited: &mut std::collections::HashSet<Address>,
    ) -> Result<()> {
        if hops_remaining == 0 {
            return Ok(());
        }

        visited.insert(current_token);

        for pool in self.pools.values() {
            let (next_token, fee) = if pool.token0 == current_token {
                (pool.token1, pool.fee)
            } else if pool.token1 == current_token {
                (pool.token0, pool.fee)
            } else {
                continue;
            };

            if visited.contains(&next_token) {
                continue;
            }

            current_path.pools.push(pool.address);
            current_path.tokens.push(next_token);
            current_path.fees.push(fee);

            if next_token == token_out {
                paths.push(current_path.clone());
            } else {
                self.find_paths(
                    next_token,
                    token_out,
                    hops_remaining - 1,
                    current_path,
                    paths,
                    visited,
                )?;
            }

            current_path.pools.pop();
            current_path.tokens.pop();
            current_path.fees.pop();
        }

        visited.remove(&current_token);
        Ok(())
    }

    /// Calculates the price impact of a swap
    pub fn calculate_price_impact(
        &self,
        path: &SwapPath,
        amount_in: U256,
        amount_out: U256,
    ) -> Result<f64> {
        if path.pools.is_empty() {
            return Ok(0.0);
        }

        let first_pool = self.pools.get(&path.pools[0])
            .ok_or(PanopticError::PoolNotFound)?;
        
        let last_pool = self.pools.get(path.pools.last().unwrap())
            .ok_or(PanopticError::PoolNotFound)?;

        let initial_price = first_pool.sqrt_price_x96;
        let final_price = last_pool.sqrt_price_x96;

        let price_ratio = final_price.as_u128() as f64 / initial_price.as_u128() as f64;
        let amount_ratio = amount_out.as_u128() as f64 / amount_in.as_u128() as f64;

        let impact = 1.0 - (amount_ratio / price_ratio);
        Ok(impact)
    }

    /// Gets the current price for a token pair across all fee tiers
    pub fn get_best_price(
        &self,
        token_in: Address,
        token_out: Address,
        amount: U256,
    ) -> Result<(u32, U256)> {
        let mut best_price = U256::zero();
        let mut best_fee = 0u32;

        for &fee in &self.fee_tiers {
            let pools: Vec<_> = self.pools.values()
                .filter(|p| {
                    (p.token0 == token_in && p.token1 == token_out ||
                     p.token0 == token_out && p.token1 == token_in) &&
                    p.fee == fee
                })
                .collect();

            for pool in pools {
                let zero_for_one = pool.token0 == token_in;
                let params = SwapParams {
                    zero_for_one,
                    amount_specified: amount.as_u128() as i128,
                    sqrt_price_limit_x96: if zero_for_one {
                        U256::from(4295128739)
                    } else {
                        U256::from(1461446703485210103287273052203988822378723970342)
                    },
                };

                if let Ok((_, amount_out, _, _)) = pool.simulate_swap(params) {
                    let out_amount = U256::from(amount_out.unsigned_abs());
                    if out_amount > best_price {
                        best_price = out_amount;
                        best_fee = fee;
                    }
                }
            }
        }

        if best_price == U256::zero() {
            return Err(PanopticError::NoLiquidity);
        }

        Ok((best_fee, best_price))
    }
}
