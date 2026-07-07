use std::time::{SystemTime, UNIX_EPOCH};

/// Return the current UTC time as an ISO 8601 string.
///
/// Format: `2025-03-15T14:30:00.123456789Z` (variable nanosecond precision).
pub fn now_utc() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let nanos = now.subsec_nanos();

    let (year, month, day, hour, minute, second) = secs_to_datetime(secs);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{nanos:09}Z")
}

/// Convert Unix seconds to (year, month, day, hour, minute, second) in UTC.
fn secs_to_datetime(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hour = (time_secs / 3600) as u32;
    let minute = ((time_secs % 3600) / 60) as u32;
    let second = (time_secs % 60) as u32;

    let (y, m, d) = days_to_date(days as i64);
    (y, m, d, hour, minute, second)
}

/// Convert days since 1970-01-01 to (year, month, day).
///
/// Uses Howard Hinnant's "chrono-Compatible Low-Level Date Algorithms".
/// See: https://howardhinnant.github.io/date_algorithms.html
fn days_to_date(days: i64) -> (i64, u32, u32) {
    // Shift epoch from 1970-01-01 to 0000-03-01.
    let z = days + 719468;

    let era = z.div_euclid(146097);
    let doe = z - era * 146097; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // year of era [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year [0, 365]
    let mp = (5 * doy + 2) / 153; // month phase [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // day [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // month [1, 12]
    let y = if m <= 2 { y + 1 } else { y }; // year

    (y, m as u32, d as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_now_utc_format() {
        let ts = now_utc();
        assert!(ts.ends_with('Z'), "timestamp should end with Z: {ts}");
        assert!(ts.contains('T'), "timestamp should contain T: {ts}");
    }

    #[test]
    fn test_known_epoch() {
        // 1970-01-01T00:00:00Z
        let (y, m, d, h, min, s) = secs_to_datetime(0);
        assert_eq!((y, m, d, h, min, s), (1970, 1, 1, 0, 0, 0));

        // 2024-01-01T00:00:00Z
        let (y, m, d, h, min, s) = secs_to_datetime(1704067200);
        assert_eq!((y, m, d, h, min, s), (2024, 1, 1, 0, 0, 0));
    }

    #[test]
    fn test_leap_year() {
        // 2024-03-01T00:00:00Z
        let (y, m, d, _, _, _) = secs_to_datetime(1709251200);
        assert_eq!((y, m, d), (2024, 3, 1));
    }

    #[test]
    fn test_mid_summer() {
        // 2026-07-07T12:30:00Z
        let (y, m, d, h, min, s) = secs_to_datetime(1783427400);
        assert_eq!(y, 2026);
        assert_eq!(m, 7);
        assert_eq!(d, 7);
        assert_eq!(h, 12);
        assert_eq!(min, 30);
        assert_eq!(s, 0);
    }
}
