//! Days since the Unix epoch to a civil date and back, for `strftime_now`.

// The calendar arithmetic below counts years in 400-year eras of 146 097 days, starting each year in
// March so the leap day falls last; 719 468 days separate 0000-03-01 from the Unix epoch.
// https://howardhinnant.github.io/date_algorithms.html
const DAYS_PER_ERA: i64 = 146_097; /* https://howardhinnant.github.io/date_algorithms.html */
const YEARS_PER_ERA: i64 = 400; /* https://howardhinnant.github.io/date_algorithms.html */
const EPOCH_OFFSET_DAYS: i64 = 719_468; /* https://howardhinnant.github.io/date_algorithms.html */
const MARCH: i64 = 3; /* https://howardhinnant.github.io/date_algorithms.html */
const MONTHS_PER_YEAR: i64 = 12; /* https://howardhinnant.github.io/date_algorithms.html */
const MONTHS_FROM_MARCH_TO_DECEMBER: i64 = 10; /* https://howardhinnant.github.io/date_algorithms.html */

/// Days since 1970-01-01 to a civil date, and back. Howard Hinnant's
/// algorithm, exact for every year this will ever be handed.
pub(super) fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let shifted = days + EPOCH_OFFSET_DAYS;
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted - (DAYS_PER_ERA - 1)
    } / DAYS_PER_ERA;
    let day_of_era = shifted - era * DAYS_PER_ERA;
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36_524 /* https://howardhinnant.github.io/date_algorithms.html */
        - day_of_era / (DAYS_PER_ERA - 1))
        / 365; /* https://howardhinnant.github.io/date_algorithms.html */
    let year = year_of_era + era * YEARS_PER_ERA;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100); /* https://howardhinnant.github.io/date_algorithms.html */
    let shifted_month = (5 * day_of_year + 2) / 153; /* https://howardhinnant.github.io/date_algorithms.html */
    let day = (day_of_year - (153 * shifted_month + 2) / 5 + 1) as u32; /* https://howardhinnant.github.io/date_algorithms.html */
    let month = if shifted_month < MONTHS_FROM_MARCH_TO_DECEMBER {
        shifted_month + MARCH
    } else {
        shifted_month - (MONTHS_PER_YEAR - MARCH)
    } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day) /* https://howardhinnant.github.io/date_algorithms.html */
}

pub(super) fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let year = if month <= 2 { year - 1 } else { year }; /* https://howardhinnant.github.io/date_algorithms.html */
    let era = if year >= 0 {
        year
    } else {
        year - (YEARS_PER_ERA - 1)
    } / YEARS_PER_ERA;
    let year_of_era = year - era * YEARS_PER_ERA;
    let shifted_month = if month > 2 { /* https://howardhinnant.github.io/date_algorithms.html */
        month as i64 - MARCH
    } else {
        month as i64 + (MONTHS_PER_YEAR - MARCH)
    };
    let day_of_year = (153 * shifted_month + 2) / 5 + day as i64 - 1; /* https://howardhinnant.github.io/date_algorithms.html */
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year; /* https://howardhinnant.github.io/date_algorithms.html */
    era * DAYS_PER_ERA + day_of_era - EPOCH_OFFSET_DAYS
}
