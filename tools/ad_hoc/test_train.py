from src.sera.twin.data_loader import DataLoader
from src.sera.twin.model_trainer import ModelTrainer
from pathlib import Path

DATA_DIR = Path("data")
loader = DataLoader(DATA_DIR)

indicators = {
    "population": ("demographic", 1),
    "income": ("demographic", 1),
    "unemployment_rate": ("labor", -1),
    "life_expectancy": ("social_well_being", 1),
    "school_enrollment": ("education", 1),
    "gdp_per_capita": ("economic", 1),
}

parameters = {
    "income_tax_rate": "annual_parameters",
    "education_spending_allocation": "annual_parameters",
    "healthcare_spending_allocation": "annual_parameters",
}

print("Loading data for", len(indicators), "indicators...")
indicators_df, parameters_df = loader.prepare_training_data(indicators, parameters, 2001, 2025)
print("Indicators DF shape:", indicators_df.shape)
print()

print("Training all indicators...")
trainer = ModelTrainer()
metrics = trainer.train_all_indicators(indicators_df, parameters_df, test_size=0.2)
print("Trained models:", len(trainer.models))
for ind in sorted(metrics.keys()):
    m = metrics[ind]
    if "r2_score" in m:
        print("  {}: R2={:.3f}".format(ind, m["r2_score"]))
    else:
        print("  {}: No metrics".format(ind))
