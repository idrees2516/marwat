use ethers::types::U256;
use std::cmp::{min, max};
use statrs::statistics::Statistics;

pub struct TickMath {
    min_tick: i32,
    max_tick: i32,
    tick_spacing: i32,
    sqrt_ratio_at_tick: [U256; 256],
}

impl TickMath {
    pub fn new(min_tick: i32, max_tick: i32, tick_spacing: i32) -> Self {
        let mut sqrt_ratio_at_tick = [U256::zero(); 256];
        Self::initialize_sqrt_ratios(&mut sqrt_ratio_at_tick);
        
        Self {
            min_tick,
            max_tick,
            tick_spacing,
            sqrt_ratio_at_tick,
        }
    }

    pub fn get_sqrt_ratio_at_tick(&self, tick: i32) -> Result<U256, &'static str> {
        if tick < self.min_tick || tick > self.max_tick {
            return Err("Tick out of range");
        }

        let abs_tick = tick.abs() as usize;
        let index = abs_tick >> 8;
        let remainder = abs_tick & 0xFF;

        let base = self.sqrt_ratio_at_tick[index];
        let multiplier = if remainder > 0 {
            self.calculate_sqrt_ratio_multiplier(remainder)
        } else {
            U256::from(1)
        };

        let sqrt_ratio = if tick >= 0 {
            base.saturating_mul(multiplier)
        } else {
            U256::max_value()
                .saturating_div(base)
                .saturating_div(multiplier)
        };

        Ok(sqrt_ratio)
    }

    pub fn get_tick_at_sqrt_ratio(&self, sqrt_ratio: U256) -> Result<i32, &'static str> {
        if sqrt_ratio < self.get_min_sqrt_ratio()? || sqrt_ratio > self.get_max_sqrt_ratio()? {
            return Err("SqrtRatio out of range");
        }

        let mut tick = self.binary_search_tick(sqrt_ratio)?;
        
        // Ensure tick is aligned with spacing
        tick = (tick / self.tick_spacing) * self.tick_spacing;
        
        Ok(tick)
    }

    pub fn get_next_initialized_tick(&self, tick: i32, tick_bitmap: &[u256], lte: bool) -> Result<i32, &'static str> {
        let compressed = self.compress_tick(tick)?;
        let word_pos = (compressed >> 8) as usize;
        let bit_pos = compressed & 0xFF;

        if lte {
            // Search for the nearest initialized tick less than or equal to the input tick
            if word_pos >= tick_bitmap.len() {
                return Err("Word position out of range");
            }

            let mut word = tick_bitmap[word_pos];
            word &= (U256::one() << bit_pos) - U256::one();

            if word == U256::zero() {
                // Search in previous words
                for i in (0..word_pos).rev() {
                    if tick_bitmap[i] != U256::zero() {
                        return Ok(self.decompress_tick((i << 8) + tick_bitmap[i].leading_zeros() as i32)?);
                    }
                }
                return Ok(self.min_tick);
            }

            return Ok(self.decompress_tick((word_pos << 8) + word.leading_zeros() as i32)?);
        } else {
            // Search for the nearest initialized tick greater than the input tick
            if word_pos >= tick_bitmap.len() {
                return Ok(self.max_tick);
            }

            let mut word = tick_bitmap[word_pos];
            word &= !(U256::one() << bit_pos) - U256::one();

            if word == U256::zero() {
                // Search in subsequent words
                for i in (word_pos + 1)..tick_bitmap.len() {
                    if tick_bitmap[i] != U256::zero() {
                        return Ok(self.decompress_tick((i << 8) + tick_bitmap[i].trailing_zeros() as i32)?);
                    }
                }
                return Ok(self.max_tick);
            }

            return Ok(self.decompress_tick((word_pos << 8) + word.trailing_zeros() as i32)?);
        }
    }

    pub fn get_fee_growth_inside(
        &self,
        tick_lower: i32,
        tick_upper: i32,
        tick_current: i32,
        fee_growth_global: U256,
        tick_fee_growth: &HashMap<i32, (U256, U256)>,
    ) -> Result<U256, &'static str> {
        if tick_lower > tick_upper {
            return Err("Invalid tick range");
        }

        let (lower_fee_outside, lower_fee_inside) = tick_fee_growth.get(&tick_lower)
            .copied()
            .unwrap_or((U256::zero(), U256::zero()));

        let (upper_fee_outside, upper_fee_inside) = tick_fee_growth.get(&tick_upper)
            .copied()
            .unwrap_or((U256::zero(), U256::zero()));

        let fee_growth = if tick_current < tick_lower {
            fee_growth_global.saturating_sub(lower_fee_outside)
        } else if tick_current < tick_upper {
            lower_fee_inside.saturating_add(
                fee_growth_global.saturating_sub(upper_fee_outside)
            )
        } else {
            lower_fee_inside.saturating_add(upper_fee_inside)
        };

        Ok(fee_growth)
    }

    fn initialize_sqrt_ratios(sqrt_ratio_at_tick: &mut [U256; 256]) {
        // Implementation of sqrt ratio initialization
        // This would populate the lookup table for sqrt price ratios
        // The actual implementation would use precise mathematical calculations
        for i in 0..256 {
            sqrt_ratio_at_tick[i] = Self::calculate_base_sqrt_ratio(i);
        }
    }

    fn calculate_base_sqrt_ratio(tick_index: usize) -> U256 {
        // Implementation of base sqrt ratio calculation
        // This would use precise mathematical formulas to calculate sqrt ratios
        // Placeholder implementation
        U256::from(1_000_000_000).saturating_add(U256::from(tick_index * 100))
    }

    fn calculate_sqrt_ratio_multiplier(&self, remainder: usize) -> U256 {
        // Implementation of sqrt ratio multiplier calculation
        // This would handle the fine-grained adjustments between base sqrt ratios
        // Placeholder implementation
        U256::from(1_000_000_000).saturating_add(U256::from(remainder))
    }

    fn binary_search_tick(&self, sqrt_ratio: U256) -> Result<i32, &'static str> {
        let mut low = self.min_tick;
        let mut high = self.max_tick;

        while low <= high {
            let mid = (low + high) / 2;
            let mid_sqrt_ratio = self.get_sqrt_ratio_at_tick(mid)?;

            if mid_sqrt_ratio == sqrt_ratio {
                return Ok(mid);
            } else if mid_sqrt_ratio < sqrt_ratio {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }

        Ok(high) // Return the largest tick with sqrt_ratio <= target
    }

    fn compress_tick(&self, tick: i32) -> Result<i32, &'static str> {
        if tick % self.tick_spacing != 0 {
            return Err("Tick not aligned with spacing");
        }
        Ok(tick / self.tick_spacing)
    }

    fn decompress_tick(&self, compressed: i32) -> Result<i32, &'static str> {
        let tick = compressed * self.tick_spacing;
        if tick < self.min_tick || tick > self.max_tick {
            return Err("Decompressed tick out of range");
        }
        Ok(tick)
    }

    pub fn get_min_sqrt_ratio(&self) -> Result<U256, &'static str> {
        self.get_sqrt_ratio_at_tick(self.min_tick)
    }

    pub fn get_max_sqrt_ratio(&self) -> Result<U256, &'static str> {
        self.get_sqrt_ratio_at_tick(self.max_tick)
    }

    pub fn validate_tick_range(&self, tick_lower: i32, tick_upper: i32) -> Result<(), &'static str> {
        if tick_lower >= tick_upper {
            return Err("Lower tick must be less than upper tick");
        }
        if tick_lower < self.min_tick {
            return Err("Lower tick too small");
        }
        if tick_upper > self.max_tick {
            return Err("Upper tick too large");
        }
        if tick_lower % self.tick_spacing != 0 {
            return Err("Lower tick not aligned with spacing");
        }
        if tick_upper % self.tick_spacing != 0 {
            return Err("Upper tick not aligned with spacing");
        }
        Ok(())
    }

    pub fn calculate_amount_delta(
        &self,
        tick_lower: i32,
        tick_upper: i32,
        liquidity_delta: i128,
    ) -> Result<(U256, U256), &'static str> {
        self.validate_tick_range(tick_lower, tick_upper)?;

        let sqrt_ratio_a = self.get_sqrt_ratio_at_tick(tick_lower)?;
        let sqrt_ratio_b = self.get_sqrt_ratio_at_tick(tick_upper)?;

        let (sqrt_ratio_low, sqrt_ratio_high) = if sqrt_ratio_a <= sqrt_ratio_b {
            (sqrt_ratio_a, sqrt_ratio_b)
        } else {
            (sqrt_ratio_b, sqrt_ratio_a)
        };

        let amount0 = if liquidity_delta > 0 {
            self.calculate_amount0_delta(sqrt_ratio_low, sqrt_ratio_high, liquidity_delta)?
        } else {
            self.calculate_amount0_delta(sqrt_ratio_low, sqrt_ratio_high, -liquidity_delta)?
                .neg()
        };

        let amount1 = if liquidity_delta > 0 {
            self.calculate_amount1_delta(sqrt_ratio_low, sqrt_ratio_high, liquidity_delta)?
        } else {
            self.calculate_amount1_delta(sqrt_ratio_low, sqrt_ratio_high, -liquidity_delta)?
                .neg()
        };

        Ok((amount0, amount1))
    }

    fn calculate_amount0_delta(
        &self,
        sqrt_ratio_low: U256,
        sqrt_ratio_high: U256,
        liquidity: i128,
    ) -> Result<U256, &'static str> {
        // Implementation of amount0 delta calculation
        // This would use precise mathematical formulas
        // Placeholder implementation
        Ok(U256::from(liquidity as u128))
    }

    fn calculate_amount1_delta(
        &self,
        sqrt_ratio_low: U256,
        sqrt_ratio_high: U256,
        liquidity: i128,
    ) -> Result<U256, &'static str> {
        // Implementation of amount1 delta calculation
        // This would use precise mathematical formulas
        // Placeholder implementation
        Ok(U256::from(liquidity as u128))
    }
}
