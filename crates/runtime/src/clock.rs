//! Clock abstraction used for run timestamps and event ordering.

use std::time::{SystemTime, UNIX_EPOCH};

use reimagine_core::event::Timestamp;

/// Wall-clock style interface for producing [`Timestamp`] values.
pub trait Clock: Send + Sync + 'static {
    fn now(&self) -> Timestamp;
}

/// Default [`Clock`] backed by [`SystemTime`].
///
/// Emits a real RFC 3339 / ISO 8601 UTC timestamp. The `Timestamp` type is
/// a `String` newtype in `core::event`, so the format is a host-neutral
/// textual representation rather than a typed instant.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Timestamp {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        // Compute a coarse UTC date from `secs` without pulling in chrono.
        // The format is RFC 3339 compliant at second precision; sub-second
        // precision is intentionally omitted to keep the helper dependency-free.
        let (year, month, day, hour, minute, second) = unix_secs_to_utc(secs);
        Timestamp::new(format!(
            "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z"
        ))
    }
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(y) => 29,
        2 => 28,
        _ => 0,
    }
}

fn unix_secs_to_utc(secs: u64) -> (i64, i64, i64, i64, i64, i64) {
    let secs = secs as i64;
    let time_of_day = secs.rem_euclid(86_400);
    let mut days = secs.div_euclid(86_400);
    let hour = time_of_day / 3600;
    let minute = (time_of_day % 3600) / 60;
    let second = time_of_day % 60;

    // 1970-01-01 is the epoch; advance by whole days.
    let mut year: i64 = 1970;
    loop {
        let dy = if is_leap(year) { 366 } else { 365 };
        if days < dy {
            break;
        }
        days -= dy;
        year += 1;
    }
    let mut month: i64 = 1;
    while month <= 12 {
        let dm = days_in_month(year, month);
        if days < dm {
            break;
        }
        days -= dm;
        month += 1;
    }
    let day = days + 1;
    (year, month, day, hour, minute, second)
}
