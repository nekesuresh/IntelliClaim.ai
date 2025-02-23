import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Define number of rows for training and testing
n_train = 100000
n_test = 20000

# ----- Mapping Functions for Categorical Variables -----
def map_land_surface_condition(val):
    mapping = {'n': 0.3, 'o': 0.5, 't': 0.8}
    return mapping.get(val, 0.5)

def map_foundation_type(val):
    mapping = {'h': 0.2, 'i': 0.3, 'r': 0.7, 'u': 0.8, 'w': 0.6}
    return mapping.get(val, 0.5)

def map_roof_type(val):
    mapping = {'n': 0.3, 'q': 0.6, 'x': 0.8}
    return mapping.get(val, 0.5)

def map_ground_floor_type(val):
    mapping = {'f': 0.3, 'm': 0.5, 'v': 0.6, 'x': 0.7, 'z': 0.8}
    return mapping.get(val, 0.5)

def map_other_floor_type(val):
    mapping = {'j': 0.3, 'q': 0.5, 's': 0.6, 'x': 0.7}
    return mapping.get(val, 0.5)

def map_position(val):
    mapping = {'j': 0.4, 'o': 0.5, 's': 0.6, 't': 0.3}
    return mapping.get(val, 0.5)

def map_plan_configuration(val):
    mapping = {'a': 0.4, 'c': 0.3, 'd': 0.5, 'f': 0.6, 'm': 0.4, 
               'n': 0.5, 'o': 0.3, 'q': 0.7, 's': 0.6, 'u': 0.8}
    return mapping.get(val, 0.5)

def map_legal_ownership_status(val):
    mapping = {'a': 0.3, 'r': 0.5, 'v': 0.6, 'w': 0.7}
    return mapping.get(val, 0.5)

# ----- New Mapping Functions for Aggregated Risk -----
def map_superstructure(row):
    risk = (2.0 * row["has_superstructure_adobe_mud"] +
            1.8 * row["has_superstructure_mud_mortar_stone"] +
            1.0 * row["has_superstructure_stone_flag"] +
            0.5 * row["has_superstructure_cement_mortar_stone"] +
            1.7 * row["has_superstructure_mud_mortar_brick"] +
            0.7 * row["has_superstructure_cement_mortar_brick"] +
            1.8 * row["has_superstructure_timber"] +
            2.0 * row["has_superstructure_bamboo"] +
            1.5 * row["has_superstructure_rc_non_engineered"] +
            (-1.0) * row["has_superstructure_rc_engineered"] +
            1.0 * row["has_superstructure_other"])
    return risk

def map_secondary_use(row):
    risk = (0.5 * row["has_secondary_use"] +
            0.4 * row["has_secondary_use_agriculture"] +
            0.6 * row["has_secondary_use_hotel"] +
            0.5 * row["has_secondary_use_rental"] +
            0.5 * row["has_secondary_use_institution"] +
            0.3 * row["has_secondary_se_school"] +
            0.4 * row["has_secondary_use_industry"] +
            0.4 * row["has_secondary_use_health_post"] +
            0.4 * row["has_secondary_use_gov_office"] +
            0.4 * row["has_secondary_use_use_police"] +
            0.4 * row["has_secondary_use_other"])
    return risk

def map_count_families(count):
    return (count - 1) / 9.0

# ----- Data Generation Function with Further Adjustments -----
def generate_data(n_rows, include_target=True):
    df = pd.DataFrame({
        "geo_level_1_id": np.random.randint(0, 31, n_rows),
        "geo_level_2_id": np.random.randint(0, 1428, n_rows),
        "geo_level_3_id": np.random.randint(0, 12568, n_rows),
        "count_floors_pre_eq": np.random.randint(1, 11, n_rows),
        "age": np.random.randint(1, 101, n_rows),
        "area_percentage": np.random.randint(0, 101, n_rows),
        "height_percentage": np.random.randint(0, 101, n_rows),
        "land_surface_condition": np.random.choice(['n', 'o', 't'], n_rows),
        "foundation_type": np.random.choice(['h', 'i', 'r', 'u', 'w'], n_rows),
        "roof_type": np.random.choice(['n', 'q', 'x'], n_rows),
        "ground_floor_type": np.random.choice(['f', 'm', 'v', 'x', 'z'], n_rows),
        "other_floor_type": np.random.choice(['j', 'q', 's', 'x'], n_rows),
        "position": np.random.choice(['j', 'o', 's', 't'], n_rows),
        "plan_configuration": np.random.choice(['a', 'c', 'd', 'f', 'm', 'n', 'o', 'q', 's', 'u'], n_rows),
        "has_superstructure_adobe_mud": np.random.randint(0, 2, n_rows),
        "has_superstructure_mud_mortar_stone": np.random.randint(0, 2, n_rows),
        "has_superstructure_stone_flag": np.random.randint(0, 2, n_rows),
        "has_superstructure_cement_mortar_stone": np.random.randint(0, 2, n_rows),
        "has_superstructure_mud_mortar_brick": np.random.randint(0, 2, n_rows),
        "has_superstructure_cement_mortar_brick": np.random.randint(0, 2, n_rows),
        "has_superstructure_timber": np.random.randint(0, 2, n_rows),
        "has_superstructure_bamboo": np.random.randint(0, 2, n_rows),
        "has_superstructure_rc_non_engineered": np.random.randint(0, 2, n_rows),
        "has_superstructure_rc_engineered": np.random.randint(0, 2, n_rows),
        "has_superstructure_other": np.random.randint(0, 2, n_rows),
        "legal_ownership_status": np.random.choice(['a', 'r', 'v', 'w'], n_rows),
        "count_families": np.random.randint(1, 11, n_rows),
        "has_secondary_use": np.random.randint(0, 2, n_rows),
        "has_secondary_use_agriculture": np.random.randint(0, 2, n_rows),
        "has_secondary_use_hotel": np.random.randint(0, 2, n_rows),
        "has_secondary_use_rental": np.random.randint(0, 2, n_rows),
        "has_secondary_use_institution": np.random.randint(0, 2, n_rows),
        "has_secondary_se_school": np.random.randint(0, 2, n_rows),
        "has_secondary_use_industry": np.random.randint(0, 2, n_rows),
        "has_secondary_use_health_post": np.random.randint(0, 2, n_rows),
        "has_secondary_use_gov_office": np.random.randint(0, 2, n_rows),
        "has_secondary_use_use_police": np.random.randint(0, 2, n_rows),
        "has_secondary_use_other": np.random.randint(0, 2, n_rows)
    })
    
    if include_target:
        # Map categorical features to risk scores
        land_risk = df['land_surface_condition'].apply(map_land_surface_condition)
        foundation_risk = df['foundation_type'].apply(map_foundation_type)
        roof_risk = df['roof_type'].apply(map_roof_type)
        ground_floor_risk = df['ground_floor_type'].apply(map_ground_floor_type)
        other_floor_risk = df['other_floor_type'].apply(map_other_floor_type)
        position_risk = df['position'].apply(map_position)
        plan_risk = df['plan_configuration'].apply(map_plan_configuration)
        ownership_risk = df['legal_ownership_status'].apply(map_legal_ownership_status)
        
        # Aggregated risks (scaled down by 0.7)
        superstructure_risk = df.apply(map_superstructure, axis=1) * 0.7
        secondary_use_risk = df.apply(map_secondary_use, axis=1) * 0.7
        
        # Map count_families to a normalized risk score
        families_risk = map_count_families(df["count_families"])
        
        # Further adjusted weights for numerical features
        intercept = -15.0  # Lower intercept for lower risk scores
        w_age = 0.02
        w_floors = 0.3
        w_height = 0.01
        w_count_families = 1.0  # Reduced weight on count_families
        
        # Weights for categorical risk factors (unchanged)
        w_foundation = 2.5
        w_roof = 2.0
        w_land = 1.5
        w_ground = 1.0
        w_other_floor = 0.5
        w_position = 1.0
        w_plan = 1.5
        w_ownership = 1.0
        
        # Calculate overall risk score as a linear combination of factors
        risk_score = (intercept +
                      w_age * df["age"] +
                      w_floors * df["count_floors_pre_eq"] +
                      w_height * df["height_percentage"] +
                      w_count_families * families_risk +
                      w_foundation * foundation_risk +
                      w_roof * roof_risk +
                      w_land * land_risk +
                      w_ground * ground_floor_risk +
                      w_other_floor * other_floor_risk +
                      w_position * position_risk +
                      w_plan * plan_risk +
                      w_ownership * ownership_risk +
                      superstructure_risk +
                      secondary_use_risk
                     )
        
        # Convert risk score to probability via logistic function
        prob = 1 / (1 + np.exp(-risk_score))
        df["Category"] = np.random.binomial(1, prob)
    
    return df

# ----- Generate and Save the Datasets -----
train_df = generate_data(n_train, include_target=True)
test_df = generate_data(n_test, include_target=False)

train_df.to_csv("training_data.csv", index=False)
test_df.to_csv("testing_data.csv", index=False)

submission_df = pd.DataFrame({
    "building_id": np.arange(1, n_test + 1),
    "Category": 1  # Placeholder; replace with model predictions in practice.
})
submission_df.to_csv("submission_format.csv", index=False)

print("Updated synthetic dataset generated: 100,000 training rows and 20,000 testing rows with a more balanced target distribution.")
