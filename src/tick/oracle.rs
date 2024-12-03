use ethers::types::U256;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct TickOracle {
    tick_spacing: i32,
    observations: RwLock<HashMap<i32, TickObservation>>,
    time_weighted_averages: RwLock<HashMap<i32, TimeWeightedAverage>>,
}

#[derive(Clone, Debug)]
pub struct TickObservation {
    pub timestamp: u32,
    pub tick_cumulative: U256,
    pub seconds_per_liquidity_cumulative: U256,
    pub initialized: bool,
}

#[derive(Clone, Debug)]
pub struct TimeWeightedAverage {
    pub tick: i32,
    pub liquidity: U256,
    pub timestamp: u32,
}

impl TickOracle {
    pub fn new(tick_spacing: i32) -> Self {
        Self {
            tick_spacing,
            observations: RwLock::new(HashMap::new()),
            time_weighted_averages: RwLock::new(HashMap::new()),
        }
    }

    pub async fn update_tick(
        &self,
        tick: i32,
        current_tick: i32,
        current_time: u32,
        current_liquidity: U256,
    ) -> Result<(), &'static str> {
        if tick % self.tick_spacing != 0 {
            return Err("Tick not aligned with spacing");
        }

        let mut observations = self.observations.write().await;
        let mut time_weighted_averages = self.time_weighted_averages.write().await;

        // Update observation
        observations.insert(tick, TickObservation {
            timestamp: current_time,
            tick_cumulative: U256::from(current_tick as u128)
                .saturating_mul(U256::from(current_time as u128)),
            seconds_per_liquidity_cumulative: if current_liquidity > U256::zero() {
                U256::from(current_time as u128)
                    .saturating_div(current_liquidity)
            } else {
                U256::zero()
            },
            initialized: true,
        });

        // Update time-weighted average
        time_weighted_averages.insert(tick, TimeWeightedAverage {
            tick: current_tick,
            liquidity: current_liquidity,
            timestamp: current_time,
        });

        Ok(())
    }

    pub async fn cross_tick(
        &self,
        tick: i32,
        current_time: u32,
        current_liquidity: U256,
    ) -> Result<(), &'static str> {
        let mut observations = self.observations.write().await;
        
        if let Some(observation) = observations.get_mut(&tick) {
            observation.timestamp = current_time;
            observation.tick_cumulative = U256::from(tick as u128)
                .saturating_mul(U256::from(current_time as u128));
            observation.seconds_per_liquidity_cumulative = if current_liquidity > U256::zero() {
                U256::from(current_time as u128)
                    .saturating_div(current_liquidity)
            } else {
                U256::zero()
            };
        }

        Ok(())
    }

    pub async fn get_time_weighted_average_tick(
        &self,
        tick: i32,
        period: u32,
        current_time: u32,
    ) -> Result<i32, &'static str> {
        let observations = self.observations.read().await;
        
        if let Some(observation) = observations.get(&tick) {
            if !observation.initialized {
                return Err("Observation not initialized");
            }

            let time_elapsed = current_time.saturating_sub(observation.timestamp);
            if time_elapsed == 0 || time_elapsed > period {
                return Err("Invalid time period");
            }

            let tick_cumulative = observation.tick_cumulative;
            Ok((tick_cumulative.as_u128() / time_elapsed as u128) as i32)
        } else {
            Err("Tick not observed")
        }
    }

    pub async fn get_seconds_per_liquidity_inside(
        &self,
        tick_lower: i32,
        tick_upper: i32,
        current_time: u32,
    ) -> Result<U256, &'static str> {
        let observations = self.observations.read().await;
        
        let lower_observation = observations.get(&tick_lower)
            .ok_or("Lower tick not observed")?;
        let upper_observation = observations.get(&tick_upper)
            .ok_or("Upper tick not observed")?;

        if !lower_observation.initialized || !upper_observation.initialized {
            return Err("Observations not initialized");
        }

        let time_lower = current_time.saturating_sub(lower_observation.timestamp);
        let time_upper = current_time.saturating_sub(upper_observation.timestamp);

        let seconds_per_liquidity_lower = lower_observation.seconds_per_liquidity_cumulative;
        let seconds_per_liquidity_upper = upper_observation.seconds_per_liquidity_cumulative;

        Ok(seconds_per_liquidity_upper
            .saturating_sub(seconds_per_liquidity_lower)
            .saturating_mul(U256::from(time_upper.min(time_lower))))
    }

    pub async fn observe(
        &self,
        tick: i32,
        seconds_ago: Vec<u32>,
        current_time: u32,
    ) -> Result<Vec<TickObservation>, &'static str> {
        let observations = self.observations.read().await;
        
        let mut results = Vec::with_capacity(seconds_ago.len());
        
        if let Some(observation) = observations.get(&tick) {
            for &seconds in &seconds_ago {
                if seconds > current_time.saturating_sub(observation.timestamp) {
                    return Err("Observation too old");
                }
                
                results.push(observation.clone());
            }
            Ok(results)
        } else {
            Err("Tick not observed")
        }
    }

    pub async fn clear_observation(&self, tick: i32) -> Result<(), &'static str> {
        self.observations.write().await.remove(&tick);
        self.time_weighted_averages.write().await.remove(&tick);
        Ok(())
    }

    pub async fn get_latest_observation(
        &self,
        tick: i32,
    ) -> Result<Option<TickObservation>, &'static str> {
        Ok(self.observations.read().await.get(&tick).cloned())
    }
}
