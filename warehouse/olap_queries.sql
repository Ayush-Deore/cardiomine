-- ============================================================
-- CardioMine — OLAP Demonstration Queries
-- Demonstrates Roll-up, Drill-down, Slice, and Dice operations
-- ============================================================

USE cardiomine;

-- ──────────────────────────────────────────────────────────────
-- ROLL-UP: From patient level → age-group level → population
-- ──────────────────────────────────────────────────────────────

-- Level 1: Per patient (base level)
SELECT
    fp.patient_id,
    dp.age_group,
    dp.gender,
    dd.disease_status,
    fp.cholesterol
FROM fact_patient_analysis fp
JOIN dim_patient dp ON fp.patient_id = dp.patient_id
JOIN dim_disease dd ON fp.disease_id = dd.disease_id
LIMIT 20;

-- Level 2: Roll up to age group
SELECT
    dp.age_group,
    COUNT(*)                                         AS total_patients,
    SUM(dd.disease_flag)                             AS disease_count,
    ROUND(AVG(dd.disease_flag) * 100, 1)             AS disease_prevalence_pct,
    ROUND(AVG(fp.cholesterol), 1)                    AS avg_cholesterol,
    ROUND(AVG(fp.resting_bp), 1)                     AS avg_resting_bp
FROM fact_patient_analysis fp
JOIN dim_patient dp ON fp.patient_id = dp.patient_id
JOIN dim_disease dd ON fp.disease_id = dd.disease_id
GROUP BY dp.age_group
ORDER BY FIELD(dp.age_group, 'Young', 'Adult', 'Middle Age', 'Senior');

-- Level 3: Roll up to population total
SELECT
    COUNT(*)                                         AS total_patients,
    SUM(dd.disease_flag)                             AS total_disease,
    ROUND(AVG(dd.disease_flag) * 100, 1)             AS overall_prevalence_pct,
    ROUND(AVG(fp.cholesterol), 1)                    AS avg_cholesterol,
    ROUND(AVG(fp.resting_bp), 1)                     AS avg_resting_bp
FROM fact_patient_analysis fp
JOIN dim_disease dd ON fp.disease_id = dd.disease_id;


-- ──────────────────────────────────────────────────────────────
-- DRILL-DOWN: From Year → Quarter → Month → Patient Analysis
-- ──────────────────────────────────────────────────────────────

-- Year level
SELECT
    dt.year,
    COUNT(*)                             AS total_analyses,
    ROUND(AVG(dd.disease_flag) * 100, 1) AS disease_prevalence_pct
FROM fact_patient_analysis fp
JOIN dim_date dt    ON fp.date_id    = dt.date_id
JOIN dim_disease dd ON fp.disease_id = dd.disease_id
GROUP BY dt.year
ORDER BY dt.year;

-- Quarter level
SELECT
    dt.year,
    dt.quarter,
    COUNT(*)                             AS total_analyses,
    ROUND(AVG(dd.disease_flag) * 100, 1) AS disease_prevalence_pct
FROM fact_patient_analysis fp
JOIN dim_date dt    ON fp.date_id    = dt.date_id
JOIN dim_disease dd ON fp.disease_id = dd.disease_id
GROUP BY dt.year, dt.quarter
ORDER BY dt.year, dt.quarter;

-- Month level
SELECT
    dt.year,
    dt.month,
    COUNT(*)                             AS total_analyses,
    ROUND(AVG(dd.disease_flag) * 100, 1) AS disease_prevalence_pct
FROM fact_patient_analysis fp
JOIN dim_date dt    ON fp.date_id    = dt.date_id
JOIN dim_disease dd ON fp.disease_id = dd.disease_id
GROUP BY dt.year, dt.month
ORDER BY dt.year, dt.month;


-- ──────────────────────────────────────────────────────────────
-- SLICE: Fix one dimension — analyze Female patients only
-- ──────────────────────────────────────────────────────────────
SELECT
    dp.age_group,
    COUNT(*)                                         AS female_patients,
    SUM(dd.disease_flag)                             AS disease_count,
    ROUND(AVG(dd.disease_flag) * 100, 1)             AS disease_prevalence_pct,
    ROUND(AVG(fp.cholesterol), 1)                    AS avg_cholesterol
FROM fact_patient_analysis fp
JOIN dim_patient dp ON fp.patient_id = dp.patient_id
JOIN dim_disease dd ON fp.disease_id = dd.disease_id
WHERE dp.gender = 'Female'                           -- ← SLICE on gender = Female
GROUP BY dp.age_group
ORDER BY FIELD(dp.age_group, 'Young', 'Adult', 'Middle Age', 'Senior');


-- ──────────────────────────────────────────────────────────────
-- DICE: Fix multiple dimensions — Middle Age + High Cholesterol
-- ──────────────────────────────────────────────────────────────
SELECT
    dp.age_group,
    dp.gender,
    dm.cholesterol_category,
    COUNT(*)                                         AS patients,
    SUM(dd.disease_flag)                             AS disease_count,
    ROUND(AVG(dd.disease_flag) * 100, 1)             AS disease_prevalence_pct,
    ROUND(AVG(fp.max_heart_rate), 1)                 AS avg_max_heart_rate
FROM fact_patient_analysis fp
JOIN dim_patient dp  ON fp.patient_id = dp.patient_id
JOIN dim_medical dm  ON fp.medical_id = dm.medical_id
JOIN dim_disease dd  ON fp.disease_id = dd.disease_id
WHERE dp.age_group           = 'Middle Age'          -- ← DICE condition 1
  AND dm.cholesterol_category = 'High'               -- ← DICE condition 2
GROUP BY dp.age_group, dp.gender, dm.cholesterol_category
ORDER BY dp.gender;
