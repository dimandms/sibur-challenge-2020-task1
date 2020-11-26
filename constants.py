FEATURE_GASES = ['A_CH4', 'A_C2H6', 'A_C3H8', 'A_iC4H10', 'A_nC4H10', 'A_iC5H12', 'A_nC5H12', 'A_C6H14']
FEATURE_COLUMNS = ['A_rate', 'B_rate'] + FEATURE_GASES
TARGET_COLUMNS = ['B_C2H6', 'B_C3H8', 'B_iC4H10', 'B_nC4H10']
FEATURE_GASES_MASS = [f"{gas}_mass" for gas in FEATURE_GASES]
TARGET_COLUMNS_MASS = [f"{gas}_mass" for gas in TARGET_COLUMNS]