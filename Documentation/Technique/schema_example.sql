-- Exemple de schéma SQL pour ingestion
CREATE TABLE eventlog (
  case_id VARCHAR(128),
  timestamp TIMESTAMP,
  activity VARCHAR(256),
  resource VARCHAR(128),
  feature_1 FLOAT,
  feature_2 FLOAT,
  feature_3 FLOAT,
  label INT
);
