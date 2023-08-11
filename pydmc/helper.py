import pandas as pd
import numpy as np

def generate_test_data(n_samples_per_condition=1000, correct_response_rate=(0.80, 0.90)):
    np.random.seed(42)

    # Conditions with sensory and response compatibility
    conditions = [
        ("exHULU", "comp", "comp"),
        ("exHCLU", "comp", "comp"),
        ("exHULC", "incomp", "comp"),
        ("exHCLC", "incomp", "comp"),
        ("anHULU", "comp", "comp"),
        ("anHCLU", "comp", "incomp"),
        ("anHULC", "incomp", "incomp"),
        ("anHCLC", "incomp", "comp"),
    ]

    # Subjects
    subjects = [f"Subj{i}" for i in range(1, 11)]

    # Initialize data
    data = []

    # Generate data for each condition
    for condition, sens_comp, resp_comp in conditions:
        for subject in subjects:
            for _ in range(n_samples_per_condition // len(subjects)):
                # RT (Reaction Time) as a random float value
                RT = np.random.uniform(300, 1000)

                # Error with correct response rate
                correct_rate = np.random.uniform(correct_response_rate[0], correct_response_rate[1])
                Error = np.random.choice([0, 1], p=[correct_rate, 1 - correct_rate])

                data.append([subject, condition, sens_comp, resp_comp, RT, Error])

    # Create a DataFrame
    df = pd.DataFrame(data, columns=["Subject", "condition", "sens_comp", "resp_comp", "RT", "Error"])

    # Shuffle the DataFrame
    #df = df.sample(frac=1).reset_index(drop=True)

    return df
