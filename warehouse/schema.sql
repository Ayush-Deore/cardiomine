-- ============================================================
-- CardioMine Data Warehouse — Star Schema DDL
-- Run this once in your MySQL server before the ETL load step.
-- ============================================================

CREATE DATABASE IF NOT EXISTS cardiomine
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE cardiomine;

-- ──────────────────────────────────────────
-- Dimension: Patient
-- ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dim_patient (
    patient_id  INT          NOT NULL PRIMARY KEY,
    gender      VARCHAR(10)  NOT NULL,
    age_group   VARCHAR(20)  NOT NULL
);

-- ──────────────────────────────────────────
-- Dimension: Medical Characteristics
-- ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dim_medical (
    medical_id             INT          NOT NULL PRIMARY KEY AUTO_INCREMENT,
    patient_id             INT          NOT NULL,
    chest_pain_type        VARCHAR(30)  NOT NULL,
    resting_ecg            VARCHAR(50)  NOT NULL,
    exercise_angina        VARCHAR(5)   NOT NULL,
    blood_sugar_category   VARCHAR(30)  NOT NULL,
    cholesterol_category   VARCHAR(20)  NOT NULL,
    bp_category            VARCHAR(30)  NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES dim_patient(patient_id)
);

-- ──────────────────────────────────────────
-- Dimension: Disease
-- ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dim_disease (
    disease_id     INT          NOT NULL PRIMARY KEY AUTO_INCREMENT,
    disease_name   VARCHAR(50)  NOT NULL,
    disease_status VARCHAR(30)  NOT NULL,
    disease_flag   TINYINT      NOT NULL   -- 0 = No Disease, 1 = Disease
);

-- ──────────────────────────────────────────
-- Dimension: Date (synthetic, for OLAP)
-- ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dim_date (
    date_id  INT         NOT NULL PRIMARY KEY,
    date     DATE        NOT NULL,
    day      TINYINT     NOT NULL,
    month    TINYINT     NOT NULL,
    quarter  TINYINT     NOT NULL,
    year     SMALLINT    NOT NULL
);

-- ──────────────────────────────────────────
-- Fact Table: Patient Analysis
-- ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fact_patient_analysis (
    analysis_id      INT     NOT NULL PRIMARY KEY AUTO_INCREMENT,
    patient_id       INT     NOT NULL,
    date_id          INT     NOT NULL,
    medical_id       INT     NOT NULL,
    disease_id       INT     NOT NULL,

    age              FLOAT   NOT NULL,
    resting_bp       FLOAT,
    cholesterol      FLOAT,
    max_heart_rate   FLOAT,
    st_depression    FLOAT,
    risk_probability FLOAT   DEFAULT 0.0,

    FOREIGN KEY (patient_id) REFERENCES dim_patient(patient_id),
    FOREIGN KEY (date_id)    REFERENCES dim_date(date_id),
    FOREIGN KEY (medical_id) REFERENCES dim_medical(medical_id),
    FOREIGN KEY (disease_id) REFERENCES dim_disease(disease_id)
);
