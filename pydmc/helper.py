import pandas as pd
import numpy as np
from .dmc import PrmsFit

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
    subjects = [f"{i}" for i in range(1, 11)]

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
                Error = np.random.choice([1, 0], p=[correct_rate, 1 - correct_rate])

                data.append([subject, condition, sens_comp, resp_comp, RT, Error])

    # Create a DataFrame
    df = pd.DataFrame(data, columns=["Subject", "condition", "sens_comp", "resp_comp", "RT", "Error"])

    # Shuffle the DataFrame
    #df = df.sample(frac=1).reset_index(drop=True)

    return df



def sim2data(sim) -> pd.DataFrame:
    """
    Transforms the simulation data into a DataFrame suitable for the Ob class.

    Args:
        sim: A simulation object containing the data in a dictionary format.

    Returns:
        df: A pandas DataFrame with 6 columns:
            'Subject': An integer identifier for the subject (set to 1 for all rows).
            'condition': A string representing the experimental condition (e.g., "exHULU").
            'sens_comp': A string representing sensory compatibility ("comp" or "incomp").
            'resp_comp': A string representing response compatibility ("comp" or "incomp").
            'RT': A numerical value representing reaction time.
            'Error': A numerical value representing the error (e.g., 1 for correct, 0 for incorrect).

    The sensory and response compatibility are mapped according to predefined conditions.
    """
    # Conditions with sensory and response compatibility
    conditions_mapping = {
        "exHULU": ("comp", "comp"),
        "exHCLU": ("comp", "comp"),
        "exHULC": ("incomp", "comp"),
        "exHCLC": ("incomp", "comp"),
        "anHULU": ("comp", "comp"),
        "anHCLU": ("comp", "incomp"),
        "anHULC": ("incomp", "incomp"),
        "anHCLC": ("incomp", "comp"),
    }

    data_list = []
    for condition, (rt, response) in sim.data.items():
        sens_comp, resp_comp = conditions_mapping.get(condition, (None, None))
        for rt_value, response_value in zip(rt, response):
            data_list.append({
                'Subject': 1,
                'condition': condition,
                'sens_comp': sens_comp,
                'resp_comp': resp_comp,
                'RT': rt_value,
                'Error': response_value
            })

    df = pd.DataFrame(data_list)
    return df



def generate_cafs(sim) -> pd.DataFrame:
    """
    Generates Conditional Accuracy Functions (CAFs) from the given simulation data.

    Args:
        sim: A simulation object containing the CAF data.

    Returns:
        df: A pandas DataFrame with 3 columns:
            'condition': A string representing the experimental condition (e.g., "exHULU").
            'bin': An integer representing the bin number for the CAF.
            'error': A numerical value representing the error rate, calculated as 100 minus the error value.

    The function iterates through the CAF data in the simulation object and formats it into a DataFrame.
    """

    data_list = []
    for condition, vals in sim.caf.items():
        ncafs = np.arange(1, sim.n_caf + 1)
        for bin_val, err_val in zip(ncafs, sim.caf[condition]['Error']):
            data_list.append({
                'condition': condition,
                'bin': bin_val,
                'error': 100 - err_val * 100
            })

    df = pd.DataFrame(data_list)
    return df


from dataclasses import asdict

def set_best_parameters(fit_diff) -> PrmsFit:
    """
    Extracts the best parameters from the given fit object and sets them as the new starting values.

    Args:
        fit_diff: A fit object containing the result of the fitting process, including the best parameters.

    Returns:
        prmsfit_adv: A new instance of PrmsFit with the best parameters set as the starting values.
    """

    # Extract the best parameters from the fit object
    best_prms = fit_diff.res_th.prms

    # Create a new instance of PrmsFit
    prmsfit_adv = PrmsFit()

    # Convert the best parameters to a dictionary
    best_prms_dict = asdict(best_prms)

    # Set the best parameters as the new starting values
    prmsfit_adv.set_start_values(**best_prms_dict)

    return prmsfit_adv
