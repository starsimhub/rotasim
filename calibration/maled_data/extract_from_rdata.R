# Export MAL-ED calibration targets from RotaDat.RData to CSV.
#
# Run from this directory:
#   Rscript extract_from_rdata.R
#
# Reads RotaDat.RData from the user's OneDrive (not in the repo) and writes
# small per-site CSVs into the same directory as this script.
#
# Outputs (one per site for Bangladesh and Pakistan):
#   ir_by_age_symp_<site>.csv      -- symptomatic IR by age bin (cases, PT, IR per 100 person-months)
#   ir_by_age_all_<site>.csv       -- all-infection IR by age bin (for reference / sensitivity)
#   first_infection_<site>.csv     -- per-child age at first rota infection + censoring
#   pt_by_age_<site>.csv           -- person-time by age bin

RDATA_PATH <- "/Users/aliciakraay/Library/CloudStorage/OneDrive-Bill&MelindaGatesFoundation/MAL-ED/RotaDat.RData"
# Resolve output dir: works under Rscript (commandArgs) and source().
get_script_dir <- function() {
  ca <- commandArgs(trailingOnly = FALSE)
  m <- regmatches(ca, regexpr("(?<=--file=).+", ca, perl = TRUE))
  if (length(m) > 0) return(dirname(normalizePath(m[1])))
  if (!is.null(sys.frames()) && length(sys.frames()) > 0) {
    of <- try(sys.frame(1)$ofile, silent = TRUE)
    if (!inherits(of, "try-error") && !is.null(of)) return(dirname(normalizePath(of)))
  }
  getwd()
}
OUT_DIR <- get_script_dir()

SITES <- c("Bangladesh", "Pakistan")

cat("Loading", RDATA_PATH, "\n")
e <- new.env()
load(RDATA_PATH, envir = e)

# Helper: pick country column robustly (CoxDat uses the ontology-tagged name).
country_col <- function(df) {
  cands <- c("Country", "Country [ENVO_00000009]")
  hit <- intersect(cands, colnames(df))
  if (length(hit) == 0) stop("No country column found in ", deparse(substitute(df)))
  hit[1]
}

# ---- Incidence rates by age bin (symptomatic + all-infection) ----
for (site in SITES) {
  for (kind in c("symp", "all")) {
    src <- if (kind == "symp") e$IRdat_new_symp else e$IRdat_new
    cc  <- country_col(src)
    sub <- src[src[[cc]] == site, c(cc, "age_cat", "cases", "PT", "IR")]
    sub <- sub[order(factor(sub$age_cat,
                            levels = c("<6 m","6-11 m","12-23 m","24-35 m",">=36 m"))), ]
    colnames(sub)[1] <- "country"
    out <- file.path(OUT_DIR, sprintf("ir_by_age_%s_%s.csv", kind, tolower(site)))
    write.csv(sub, out, row.names = FALSE)
    cat("  wrote", out, "(", nrow(sub), "rows)\n")
  }
}

# ---- Person-time by age bin ----
for (site in SITES) {
  sub <- e$pt_summary[e$pt_summary$Country == site, ]
  sub <- sub[order(factor(sub$age_cat,
                          levels = c("<6 m","6-11 m","12-23 m","24-35 m",">=36 m"))), ]
  out <- file.path(OUT_DIR, sprintf("pt_by_age_%s.csv", tolower(site)))
  write.csv(sub, out, row.names = FALSE)
  cat("  wrote", out, "(", nrow(sub), "rows)\n")
}

# ---- First-infection survival data (CoxDat) ----
# Per-child columns needed: participant id, country, age at first event (or censoring), event indicator.
# RotaCensor: 1 = event observed, 0 = censored (verify against SummaryFirstInfect medians).
cox <- as.data.frame(e$CoxDat)
cc  <- country_col(cox)
# Derive event indicator from DateRota directly (DateRota != NA => first infection observed).
# In RotaDat.RData, RotaCensor is coded 1 = censored, 0 = event observed (verified by
# median-age sanity check against SummaryFirstInfect below).
birth_col <- "Birth date [EFO_0004950]"
cox$event_observed <- as.integer(!is.na(cox$DateRota))
cox$age_event_days <- ifelse(
  cox$event_observed == 1,
  as.integer(cox$DateRota)  - as.integer(cox[[birth_col]]),
  as.integer(cox$LastDate)  - as.integer(cox[[birth_col]])
)
cox$age_event_months <- cox$age_event_days / 30.4375

for (site in SITES) {
  sub <- cox[cox[[cc]] == site, c("Participant_ID", "Sex [PATO_0000047]",
                                  "age_event_days", "age_event_months",
                                  "event_observed")]
  colnames(sub) <- c("participant_id", "sex", "age_event_days", "age_event_months", "event_observed")
  out <- file.path(OUT_DIR, sprintf("first_infection_%s.csv", tolower(site)))
  write.csv(sub, out, row.names = FALSE)
  cat("  wrote", out, "(", nrow(sub), "rows)\n")
}

# ---- Sanity check: replicate SummaryFirstInfect medians from CoxDat ----
cat("\nSanity check: median age at first infection (events only) by country\n")
for (site in SITES) {
  sub <- cox[cox[[cc]] == site & cox$event_observed == 1, ]
  med_days <- median(sub$age_event_days, na.rm = TRUE)
  med_mos  <- median(sub$age_event_months, na.rm = TRUE)
  n_events <- nrow(sub)
  cat(sprintf("  %-12s n_events=%d  median_days=%.1f  median_months=%.2f\n",
              site, n_events, med_days, med_mos))
}
cat("\nCompare to SummaryFirstInfect:\n")
print(e$SummaryFirstInfect[e$SummaryFirstInfect$Country %in% SITES,
                           c("Country","med_age","med_age_mos","Vaccination")])

cat("\nDone.\n")
