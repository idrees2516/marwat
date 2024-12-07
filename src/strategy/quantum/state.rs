use super::types::*;
use ethers::types::{Address, U256};
use std::collections::HashMap;

#[derive(Debug)]
pub struct QuantumState {
    pub positions: HashMap<U256, LeggedPosition>,
    pub market_data: MarketData,
    pub hedge_params: HedgeParameters,
    pub last_rebalance: u64,
    pub total_value_locked: U256,
}

impl QuantumState {
    pub fn new(market_data: MarketData, hedge_params: HedgeParameters) -> Self {
        Self {
            positions: HashMap::new(),
            market_data,
            hedge_params,
            last_rebalance: 0,
            total_value_locked: U256::zero(),
        }
    }

    pub fn add_position(&mut self, token_id: U256, position: LeggedPosition) {
        self.total_value_locked = self.total_value_locked.saturating_add(position.total_collateral);
        self.positions.insert(token_id, position);
    }

    pub fn remove_position(&mut self, token_id: U256) -> Option<LeggedPosition> {
        if let Some(position) = self.positions.remove(&token_id) {
            self.total_value_locked = self.total_value_locked.saturating_sub(position.total_collateral);
            Some(position)
        } else {
            None
        }
    }

    pub fn update_market_data(&mut self, new_data: MarketData) {
        self.market_data = new_data;
    }

    pub fn needs_rebalance(&self, current_time: u64) -> bool {
        current_time.saturating_sub(self.last_rebalance) >= self.hedge_params.rebalance_interval
    }

    pub fn calculate_portfolio_greeks(&self) -> (f64, f64, f64, f64) {
        let mut total_delta = 0.0;
        let mut total_gamma = 0.0;
        let mut total_theta = 0.0;
        let mut total_vega = 0.0;

        for position in self.positions.values() {
            for leg in &position.legs {
                total_delta += leg.delta * leg.ratio as f64;
                total_gamma += leg.gamma * leg.ratio as f64;
                total_theta += leg.theta * leg.ratio as f64;
                total_vega += leg.vega * leg.ratio as f64;
            }
        }

        (total_delta, total_gamma, total_theta, total_vega)
    }

    pub fn validate_risk_parameters(&self) -> bool {
        for position in self.positions.values() {
            let (delta, gamma, theta, vega) = self.calculate_portfolio_greeks();
            let limits = &position.risk_params.greek_limits;

            if delta.abs() > limits.max_delta
                || gamma.abs() > limits.max_gamma
                || theta.abs() > limits.max_theta
                || vega.abs() > limits.max_vega
            {
                return false;
            }
        }
        true
    }
}
