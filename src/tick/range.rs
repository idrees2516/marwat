use ethers::types::U256;

#[derive(Clone, Debug)]
pub struct TickRange {
    pub tick_lower: i32,
    pub tick_upper: i32,
    pub liquidity: U256,
    pub fee_growth_inside_last: U256,
    pub seconds_per_liquidity_inside_last: U256,
    pub last_update_time: u32,
}

impl TickRange {
    pub fn new(tick_lower: i32, tick_upper: i32) -> Self {
        Self {
            tick_lower,
            tick_upper,
            liquidity: U256::zero(),
            fee_growth_inside_last: U256::zero(),
            seconds_per_liquidity_inside_last: U256::zero(),
            last_update_time: 0,
        }
    }

    pub fn update(
        &mut self,
        liquidity_delta: i128,
        current_tick: i32,
        current_time: u32,
        current_liquidity: U256,
    ) -> Result<(), &'static str> {
        // Update liquidity
        self.liquidity = if liquidity_delta >= 0 {
            self.liquidity.saturating_add(U256::from(liquidity_delta as u128))
        } else {
            self.liquidity.saturating_sub(U256::from((-liquidity_delta) as u128))
        };

        // Update time tracking
        if current_tick >= self.tick_lower && current_tick < self.tick_upper {
            let time_delta = current_time.saturating_sub(self.last_update_time);
            if current_liquidity > U256::zero() {
                self.seconds_per_liquidity_inside_last = self.seconds_per_liquidity_inside_last
                    .saturating_add(U256::from(time_delta)
                    .saturating_div(current_liquidity));
            }
        }

        self.last_update_time = current_time;

        Ok(())
    }

    pub fn is_in_range(&self, tick: i32) -> bool {
        tick >= self.tick_lower && tick < self.tick_upper
    }

    pub fn update_fees(
        &mut self,
        fee_growth_inside: U256,
        seconds_per_liquidity_inside: U256,
        current_time: u32,
    ) {
        self.fee_growth_inside_last = fee_growth_inside;
        self.seconds_per_liquidity_inside_last = seconds_per_liquidity_inside;
        self.last_update_time = current_time;
    }

    pub fn get_fees_earned(
        &self,
        fee_growth_inside: U256,
    ) -> U256 {
        fee_growth_inside
            .saturating_sub(self.fee_growth_inside_last)
            .saturating_mul(self.liquidity)
    }

    pub fn get_seconds_inside(
        &self,
        seconds_per_liquidity_inside: U256,
    ) -> U256 {
        seconds_per_liquidity_inside
            .saturating_sub(self.seconds_per_liquidity_inside_last)
            .saturating_mul(self.liquidity)
    }
}
