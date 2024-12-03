use ethers::types::U256;
use std::collections::HashMap;

pub struct TickBitmap {
    bitmap: HashMap<i16, U256>,
    tick_spacing: i32,
}

impl TickBitmap {
    pub fn new(tick_spacing: i32) -> Self {
        Self {
            bitmap: HashMap::new(),
            tick_spacing,
        }
    }

    pub fn flip_tick(&mut self, tick: i32) -> Result<(), &'static str> {
        if tick % self.tick_spacing != 0 {
            return Err("Tick not aligned with spacing");
        }

        let (word_pos, bit_pos) = self.position(tick);
        let mask = U256::one() << bit_pos;

        let word = self.bitmap.entry(word_pos).or_insert(U256::zero());
        *word ^= mask;

        Ok(())
    }

    pub fn is_initialized(&self, tick: i32) -> Result<bool, &'static str> {
        if tick % self.tick_spacing != 0 {
            return Err("Tick not aligned with spacing");
        }

        let (word_pos, bit_pos) = self.position(tick);
        let word = self.bitmap.get(&word_pos).copied().unwrap_or(U256::zero());
        
        Ok((word & (U256::one() << bit_pos)) != U256::zero())
    }

    pub fn next_initialized_tick_within_word(
        &self,
        tick: i32,
        lte: bool,
    ) -> Result<(i32, bool), &'static str> {
        let (word_pos, bit_pos) = self.position(tick);
        let word = self.bitmap.get(&word_pos).copied().unwrap_or(U256::zero());

        let compressed = tick / self.tick_spacing;
        let next_bit_pos = if lte {
            // Search for nearest initialized tick less than or equal to tick
            if word == U256::zero() {
                return Ok((compressed - bit_pos as i32, false));
            }

            let masked = word & ((U256::one() << bit_pos) - U256::one());
            if masked == U256::zero() {
                return Ok((compressed - bit_pos as i32, false));
            }

            bit_pos - 1 - masked.leading_zeros() as u8
        } else {
            // Search for nearest initialized tick greater than tick
            let masked = word & !((U256::one() << bit_pos) - U256::one());
            if masked == U256::zero() {
                return Ok((compressed + (255 - bit_pos) as i32, false));
            }

            bit_pos + masked.trailing_zeros() as u8
        };

        Ok(((word_pos as i32) * 256 + next_bit_pos as i32, true))
    }

    pub fn next_initialized_tick(
        &self,
        tick: i32,
        lte: bool,
    ) -> Result<(i32, bool), &'static str> {
        let compressed = tick / self.tick_spacing;
        let (word_pos, bit_pos) = self.position(tick);

        let (next_tick, initialized) = self.next_initialized_tick_within_word(tick, lte)?;

        if initialized {
            return Ok((next_tick * self.tick_spacing, true));
        }

        // Search in adjacent words
        let (next_word_pos, found) = self.next_initialized_word(word_pos, lte)?;
        
        if !found {
            return Ok((
                if lte { compressed - bit_pos as i32 } else { compressed + (255 - bit_pos) as i32 }
                * self.tick_spacing,
                false
            ));
        }

        let next_word = self.bitmap.get(&next_word_pos).copied().unwrap_or(U256::zero());
        let next_bit_pos = if lte {
            255 - next_word.leading_zeros() as u8
        } else {
            next_word.trailing_zeros() as u8
        };

        Ok(((next_word_pos as i32 * 256 + next_bit_pos as i32) * self.tick_spacing, true))
    }

    fn position(&self, tick: i32) -> (i16, u8) {
        let compressed = tick / self.tick_spacing;
        let word_pos = compressed >> 8;
        let bit_pos = (compressed & 0xFF) as u8;
        (word_pos as i16, bit_pos)
    }

    fn next_initialized_word(&self, word_pos: i16, lte: bool) -> Result<(i16, bool), &'static str> {
        if lte {
            // Search for nearest initialized word less than or equal to word_pos
            for pos in (i16::MIN..word_pos).rev() {
                if let Some(word) = self.bitmap.get(&pos) {
                    if *word != U256::zero() {
                        return Ok((pos, true));
                    }
                }
            }
            Ok((word_pos, false))
        } else {
            // Search for nearest initialized word greater than word_pos
            for pos in (word_pos + 1)..i16::MAX {
                if let Some(word) = self.bitmap.get(&pos) {
                    if *word != U256::zero() {
                        return Ok((pos, true));
                    }
                }
            }
            Ok((word_pos, false))
        }
    }

    pub fn clear(&mut self) {
        self.bitmap.clear();
    }

    pub fn get_populated_words(&self) -> Vec<(i16, U256)> {
        self.bitmap
            .iter()
            .filter(|(_, &word)| word != U256::zero())
            .map(|(&pos, &word)| (pos, word))
            .collect()
    }

    pub fn get_word(&self, word_pos: i16) -> U256 {
        self.bitmap.get(&word_pos).copied().unwrap_or(U256::zero())
    }

    pub fn set_word(&mut self, word_pos: i16, word: U256) {
        if word == U256::zero() {
            self.bitmap.remove(&word_pos);
        } else {
            self.bitmap.insert(word_pos, word);
        }
    }

    pub fn validate_tick(&self, tick: i32) -> Result<(), &'static str> {
        if tick % self.tick_spacing != 0 {
            return Err("Tick not aligned with spacing");
        }
        Ok(())
    }

    pub fn count_initialized_ticks(&self) -> u32 {
        self.bitmap
            .values()
            .map(|word| word.count_ones())
            .sum()
    }

    pub fn get_initialized_ticks_in_word(&self, word_pos: i16) -> Vec<i32> {
        let word = self.get_word(word_pos);
        let mut ticks = Vec::new();

        for bit_pos in 0..256 {
            if word & (U256::one() << bit_pos) != U256::zero() {
                let tick = (word_pos as i32 * 256 + bit_pos) * self.tick_spacing;
                ticks.push(tick);
            }
        }

        ticks
    }

    pub fn get_initialized_ticks_in_range(
        &self,
        tick_lower: i32,
        tick_upper: i32,
    ) -> Result<Vec<i32>, &'static str> {
        self.validate_tick(tick_lower)?;
        self.validate_tick(tick_upper)?;

        let (lower_word, _) = self.position(tick_lower);
        let (upper_word, _) = self.position(tick_upper);

        let mut ticks = Vec::new();
        for word_pos in lower_word..=upper_word {
            ticks.extend(self.get_initialized_ticks_in_word(word_pos));
        }

        ticks.retain(|&tick| tick >= tick_lower && tick <= tick_upper);
        ticks.sort_unstable();

        Ok(ticks)
    }
}
