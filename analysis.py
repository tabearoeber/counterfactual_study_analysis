# to read in the data
import pandas as pd
import json
import numpy as np

# to plot the data
import seaborn as sns
import matplotlib.pyplot as plt

# for anova
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf

from scipy import stats

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import sys

# Save original stdout if you need to restore it later
original_stdout = sys.stdout

# # Open a file and redirect stdout to it
# sys.stdout = open('output.txt', 'w')



def load_valid_data(data_path='./data/'):
    # Load JSON file
    with open("./valid_users.json", "r") as file:
        valid_user_ids = json.load(file)

    # print(len(valid_user_ids))

    # load in responses
    df = pd.read_csv(f'{data_path}/answers.csv')
    df = df[df['userID'].isin(valid_user_ids)]

    # load demographics
    demographics = pd.read_csv(f'{data_path}/demographics.csv')
    demographics = demographics[demographics['userID'].isin(valid_user_ids)]
    demographics_short = demographics[['userID','age', 'preferredCurrency', 'education', 'gender', 'houseBuying', 
                                    'dailyAIWork', 'theoreticalKnowledge', 'practicalExperience',
                                    'attributedUserExplanationViewMode', 'attributedUserExplanationType']]

    # import house information data 
    with open('../code/explanations/utrecht-housing-rf-50Kless/rho-0.05/filtered/houses.json') as f:
        d = json.load(f)

    houses = pd.DataFrame(d)

    # get batch info data
    # batch_info1 = pd.read_csv('...') # adjust path to amazon batch info files
    # batch_info2 = pd.read_csv('...') # adjust path to amazon batch info files
    batch_info = pd.concat([batch_info1, batch_info2])

    return df, houses, demographics_short, batch_info

import pandas as pd

def disambiguate_user_ids(df):
    # Group by userID and collect all unique WorkTimeInSeconds values
    grouped = df.groupby('userID')['WorkTimeInSeconds'].nunique()

    # Identify userIDs that appear with multiple WorkTimeInSeconds values
    problematic_ids = grouped[grouped > 1].index

    # Counter to assign suffixes
    suffix_counts = {}

    # New userID column
    new_user_ids = []

    for _, row in df.iterrows():
        uid = row['userID']
        wtime = row['WorkTimeInSeconds']

        if uid in problematic_ids:
            key = (uid, wtime)
            if key not in suffix_counts:
                suffix = chr(ord('a') + len([k for k in suffix_counts if k[0] == uid]))
                suffix_counts[key] = f"{uid}_{suffix}"
            new_user_ids.append(suffix_counts[key])
        else:
            new_user_ids.append(uid)

    df['userID'] = new_user_ids
    return df

def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        outliers_iqr = (df[column] < (Q1-1.5*IQR)) | (df[column] > (Q3+1.5*IQR))
        df = df[~outliers_iqr]

    return df


def clean_data(df, demographics, houses, batch_info, phase='p0', transformation = None):
    """
    data: df with survey responses
    demographics: df with demographics
    houses: df with house info
    phase (str): one of 'p0','p1','p2'
    """
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
    from scipy.stats import boxcox

    df = remove_outliers(df, ['propertyValue', 'aiPrediction'])

    # demographics = demographics.merge(batch_info[['Answer.surveycode', 'WorkTimeInSeconds']], left_on='userID', right_on='Answer.surveycode', how='left')

    if phase=='p0':
        phase0 = df[df['phase']=='p0'] # get answers from p0 
        phase0 = phase0[['userID', 'houseId', 'propertyValue']] # choose relevant columns

        # merge data from p0 with demographics and house information
        phase0 = pd.merge(phase0, demographics, on='userID')
        phase0 = pd.merge(phase0, houses[['retailvalue', 'id']], left_on='houseId', right_on='id')
        phase0 = phase0.drop(['id'], axis=1) # remove redundant house id
        phase0 = pd.merge(phase0, batch_info[['Answer.surveycode', 'WorkTimeInSeconds', 'SubmitTime']], left_on='userID', right_on='Answer.surveycode', how='left')
        phase0 = phase0.drop(['Answer.surveycode'], axis=1) #remove redundant user id
        phase0 = disambiguate_user_ids(phase0)

        # # remove users with slowest 0.05 quantile 
        cutoff = phase0['WorkTimeInSeconds'].quantile(0.05)
        phase0 = phase0[phase0['WorkTimeInSeconds']>cutoff]

        # remove nonsensible values 
        # phase0 = phase0[(phase0['propertyValue']>200000) & (phase0['propertyValue']<2000000)]

        # calculate gap between true price and user's estimated price
        phase0['gap_house_price'] = abs(phase0['retailvalue']-phase0['propertyValue'])
        # filter data -- only keep data within xth percentile
        # phase0 = phase0[phase0['gap_house_price'] < phase0['gap_house_price'].quantile(0.85)]

        # remove outliers based on IQR
        # phase0 = remove_outliers(phase0, ['gap_house_price'])

        # standardize dependent variable
        scaler = StandardScaler()
        # phase0['gap_house_price_scaled'] = scaler.fit_transform(phase0[['gap_house_price']])
        # phase0['age_scaled'] = scaler.fit_transform(phase0[['age']])
        if transformation == 'log':
            phase0[f'{transformation}_gap_house_price'] = np.log(phase0['gap_house_price'])
            # phase0['log_gap_house_price_scaled'] = scaler.fit_transform(phase0[['log_gap_house_price']])
        elif transformation == 'sqrt':
            phase0[f'{transformation}_gap_house_price'] = np.sqrt(phase0['gap_house_price'])
            # phase0['sqrt_gap_house_price_scaled'] = scaler.fit_transform(phase0[['sqrt_gap_house_price']])
        elif transformation == 'boxcox':
            phase0[f'{transformation}_gap_house_price'] = boxcox(phase0['gap_house_price'])[0]
        
        for num in phase0.select_dtypes(include=['number']):
            phase0[num+'_scaled'] = scaler.fit_transform(phase0[[num]])

        # categorical_features = phase0.select_dtypes(include=['object'])
        # print(phase0.head())

        # merge 'PhD' and 'masters'
        # Replace 'PhD' with 'masters' in the 'education' column
        phase0['education'] = phase0['education'].replace('phd', 'masters')
        # phase0['education'] = phase0['education'].replace('highschool', 'bachelors')
        # Replace 'less_than_monthly' with 'monthly'
        phase0['dailyAIWork'] = phase0['dailyAIWork'].replace('less_than_monthly', 'monthly')

        return phase0
    
    elif phase=='p1':
        phase1 = df[df['phase']=='p1'] # get answers from p0 
        phase1 = phase1[['userID', 'houseId', 'propertyValue', 'aiPrediction']] # choose relevant columns

        # merge data from p0 with demographics and house information
        phase1 = pd.merge(phase1, demographics, on='userID')
        phase1 = pd.merge(phase1, houses[['retailvalue', 'id', 'predicted-price']], left_on='houseId', right_on='id')
        phase1 = phase1.drop(['id'], axis=1) # remove redundant house id
        phase1 = pd.merge(phase1, batch_info[['Answer.surveycode', 'WorkTimeInSeconds', 'SubmitTime']], left_on='userID', right_on='Answer.surveycode', how='left')
        phase1 = phase1.drop(['Answer.surveycode'], axis=1) #remove redundant user id
        phase1 = disambiguate_user_ids(phase1)

        # # remove users with slowest 0.05 quantile 
        cutoff = phase1['WorkTimeInSeconds'].quantile(0.05)
        phase1= phase1[phase1['WorkTimeInSeconds']>cutoff]

        # remove nonsensible values 
        # phase0 = phase0[(phase0['propertyValue']>200000) & (phase0['propertyValue']<2000000)]

        # calculate gap between true price and user's estimated price
        phase1['gap_house_price'] = abs(phase1['retailvalue']-phase1['propertyValue'])
        phase1['gap_ai_price'] = abs(phase1['predicted-price']-phase1['aiPrediction'])
        # filter data -- only keep data within xth percentile
        # phase0 = phase0[phase0['gap_house_price'] < phase0['gap_house_price'].quantile(0.85)]

        # remove outliers based on IQR
        phase1 = remove_outliers(phase1, ['gap_house_price', 'gap_ai_price'])

        # standardize dependent variable
        scaler = StandardScaler()
        # phase0['gap_house_price_scaled'] = scaler.fit_transform(phase0[['gap_house_price']])
        # phase0['age_scaled'] = scaler.fit_transform(phase0[['age']])
        if transformation == 'log':
            phase1[f'{transformation}_gap_house_price'] = np.log(phase1['gap_house_price'])
            phase1[f'{transformation}_gap_ai_price'] = np.log(phase1['gap_ai_price'])
            # phase0['log_gap_house_price_scaled'] = scaler.fit_transform(phase0[['log_gap_house_price']])
        elif transformation == 'sqrt':
            phase1[f'{transformation}_gap_house_price'] = np.sqrt(phase1['gap_house_price'])
            phase1[f'{transformation}_gap_ai_price'] = np.sqrt(phase1['gap_ai_price'])
        elif transformation == 'boxcox':
            phase1[f'{transformation}_gap_house_price'] = boxcox(phase1['gap_house_price'])[0]
            phase1[f'{transformation}_gap_ai_price'] = boxcox(phase1['gap_ai_price'])[0]
            
        
        for num in phase1.select_dtypes(include=['number']):
            phase1[num+'_scaled'] = scaler.fit_transform(phase1[[num]])

        # merge 'PhD' and 'masters'
        phase1['education'] = phase1['education'].replace('phd', 'masters')
        # Replace 'less_than_monthly' with 'monthly'
        phase1['dailyAIWork'] = phase1['dailyAIWork'].replace('less_than_monthly', 'monthly')

        return phase1
    
    elif phase=='p2':
        phase2 = df[df['phase']=='p2'] # get answers from p0 
        phase2 = phase2[['userID', 'houseId', 'propertyValue', 'followAI', 'featuresHidden']] # choose relevant columns

        # merge data from p0 with demographics and house information
        phase2 = pd.merge(phase2, demographics, on='userID')
        phase2 = pd.merge(phase2, houses[['retailvalue', 'id', 'predicted-price']], left_on='houseId', right_on='id')
        phase2 = phase2.drop(['id'], axis=1) # remove redundant house id
        phase2 = pd.merge(phase2, batch_info[['Answer.surveycode', 'WorkTimeInSeconds', 'SubmitTime']], left_on='userID', right_on='Answer.surveycode', how='left')
        phase2 = phase2.drop(['Answer.surveycode'], axis=1) #remove redundant user id
        phase2 = disambiguate_user_ids(phase2)
        
        # # remove users with slowest 0.05 quantile 
        cutoff = phase2['WorkTimeInSeconds'].quantile(0.05)
        phase2 = phase2[phase2['WorkTimeInSeconds']>cutoff]

        phase2['estimated-price'] = np.where(phase2['followAI'], phase2['predicted-price'], phase2['propertyValue'])
        phase2['abs_error_pred'] = abs(phase2['predicted-price'] - phase2['retailvalue'])

        phase_data_users = phase2.groupby('userID')['followAI'].agg(['mean']).reset_index()
        phase_data_users['mean'] = phase_data_users['mean'].astype(float)
        phase_data_users = phase_data_users.rename({'mean':'followAITendency'}, axis=1)
        phase2 = pd.merge(phase2, phase_data_users, on='userID')

        if transformation == 'log':
            phase2[f'{transformation}_abs_error_pred'] = np.log(phase2['abs_error_pred'])
        elif transformation == 'sqrt':
            phase2[f'{transformation}_abs_error_pred'] = np.sqrt(phase2['abs_error_pred'])
        elif transformation == 'boxcox':
            phase2[f'{transformation}_abs_error_pred'] = boxcox(phase2['abs_error_pred'])[0]

        scaler = StandardScaler()
        for num in phase2.select_dtypes(include=['number']):
            phase2[num+'_scaled'] = scaler.fit_transform(phase2[[num]])

        # merge 'PhD' and 'masters'
        phase2['education'] = phase2['education'].replace('phd', 'masters')
        # Replace 'less_than_monthly' with 'monthly'
        phase2['dailyAIWork'] = phase2['dailyAIWork'].replace('less_than_monthly', 'monthly')

        user_counts = phase2['userID'].value_counts()
        users_p2_complete = user_counts[user_counts >= 15].index
        # users_p2_complete = user_counts[user_counts == 30].index
        phase2 = phase2[phase2['userID'].isin(users_p2_complete)]

        return phase2
    
from scipy.stats import skew, kurtosis


def get_descriptives(df):
    print('Descriptive Statistics:\n')
    # unique_users = df.drop_duplicates(subset=['userID', 'WorkTimeInSeconds'])
    unique_users = df.drop_duplicates(subset=['userID'])
    print(f'Nr of participants: {len(unique_users)}\n')

    categorical_features = list(unique_users.select_dtypes(include=['category', 'object']).columns)
    categorical_features.remove('userID')

    for column in categorical_features:
        print(f'{unique_users[column].value_counts()}\n')

    print(pd.crosstab(unique_users['attributedUserExplanationType'],unique_users['attributedUserExplanationViewMode']))


def check_anova_assumptions(df, dependent, independent, covariates, model, model_type = None, name = None, save_plots=False):
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    if save_plots:
        if name is not None:
            if not os.path.exists(f'./assumptions-plots/{name}/'):
                os.mkdir(f'./assumptions-plots/{name}/')

    if model_type is not None:
        model_type_str = model_type.replace('-', ' ').title()
    print(f'\n--- Checking {model_type_str} ANOVA Assumptions---')
    # print("\n--- Checking ANOVA Assumptions ---")

    # ASSUMPTION 1: NORMALITY OF RESIDUALS
    w, pvalue = stats.shapiro(model.resid)
    df['residuals'] = model.resid

    # skewness and kurtosis
    sk = skew(df['residuals'])
    kurt = kurtosis(df['residuals'])  # Excess kurtosis
    print(f"Skewness: {sk:.4f}")
    print(f"Kurtosis: {kurt:.4f}")
    
    if abs(sk) <= 1.0 and abs(kurt) <= 1.0:
        approx_normal = '✅' 
    else:
        approx_normal = '❌'

    # Shapiro-Wilk
    print(f'Shapiro-Wilk test for normality: W={w:.3f}, p={pvalue:.3f}')
    shapiro_normal = '✅'  if pvalue > 0.05 else '❌'

    
    # # histogram of residuals
    # plt.figure(figsize=(6, 4))
    # plt.hist(df['residuals'], bins=30, edgecolor='black', linewidth=0.1, color='#85BAE7')
    # plt.xlabel("Residual", fontsize=15)
    # plt.ylabel("Frequency", fontsize=15)
    # plt.tick_params(axis='both', labelsize=12)  # Tick labels
    # plt.tight_layout()
    # if save_plots:
    #     basename = f'residuals-hist-{model_type}'
    #     filepath = get_next_filename(basename, directory=f'./assumptions-plots/{name}/')
    #     plt.savefig(filepath)
    # plt.show()
    # plt.close()
    # plt.figure(figsize=(6, 4))


    # Q-Q plot of residuals
    # plt.figure(figsize=(6, 4))
    # stats.probplot(df['residuals'], dist="norm", plot=plt)
    # plt.title(f" ")
    # plt.tick_params(axis='both', labelsize=12)  # Tick labels
    # plt.xlabel("Theoretical quantiles", fontsize=15)
    # plt.ylabel("Ordered values", fontsize=15)
    # plt.tight_layout()
    # if save_plots:
    #     basename = f'qq-plot-{model_type}'
    #     filepath = get_next_filename(basename, directory=f'./assumptions-plots/{name}/')
    #     plt.savefig(filepath)
    # plt.show()
    # plt.close()


    # ASSUMPTION 2: HOMOGENEITY
    # Homogeneity of variances (Levene's Test)
    grouped = [group[dependent].values for name, group in df.groupby(independent)]
    stat, p_levene = stats.levene(*grouped)
    print(f'Levene’s test for homogeneity of variances: stat={stat:.3f}, p={p_levene:.3f}')

    # ASSUMPTION 3: absence of multicollinearity
    # high_corr, upper = show_high_correlations(df[covariates], threshold=0.7)
    # if high_corr:
    #     print(f"Features with correlation above {0.7}:")
    #     for col1, col2, corr in sorted(high_corr, key=lambda x: -x[2]):
    #         print(f"{col1} and {col2}: {corr:.2f}")
    # print(upper)
    high_corr_pearson, high_corr_spearman, pearson_matrix, spearman_matrix = show_high_correlations(df[covariates], threshold=0.7)
    print('Pearson matrix:')
    print(pearson_matrix)
    print('\nSpearman matrix:')
    print(spearman_matrix)

    # # ASSUMPTION 4: homoscedasticity
    fitted_vals = model.fittedvalues
    # plt.figure(figsize=(6, 4))
    # plt.scatter(fitted_vals, df['residuals'], alpha=0.5, color='#85BAE7')
    # plt.axhline(0, linestyle='--', color='gray')
    # plt.xlabel("Fitted Values", fontsize=15)
    # plt.ylabel("Residuals", fontsize=15)
    # plt.tick_params(axis='both', labelsize=12)  # Tick labels
    # # plt.title(f"Residuals vs Fitted for {model_type_str} ANOVA \nwith {covariates} \nas predictors")
    # plt.tight_layout()
    # if save_plots:
    #     basename = f'homoscedasticity-{model_type}'
    #     filepath = get_next_filename(basename, directory=f'./assumptions-plots/{name}/')
    #     plt.savefig(filepath)
    # plt.show()
    # plt.close()


    # plot 
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))

    axs[0].hist(df['residuals'], bins=30, edgecolor='black', linewidth=0.1, color='#85BAE7')
    axs[0].set_xlabel("Residual", fontsize=15)
    axs[0].set_ylabel("Frequency", fontsize=15)
    axs[0].tick_params(axis='both', labelsize=12)  # Tick labels

    stats.probplot(df['residuals'], dist="norm", plot=axs[1])
    axs[1].set_title(" ")
    axs[1].tick_params(axis='both', labelsize=12)  # Tick labels
    axs[1].set_xlabel("Theoretical quantiles", fontsize=15)
    axs[1].set_ylabel("Ordered values", fontsize=15)

    fitted_vals = model.fittedvalues
    axs[2].scatter(fitted_vals, df['residuals'], alpha=0.5, color='#85BAE7')
    axs[2].axhline(0, linestyle='--', color='gray')
    axs[2].set_xlabel("Fitted Values", fontsize=15)
    axs[2].set_ylabel("Residuals", fontsize=15)
    axs[2].tick_params(axis='both', labelsize=12)  # Tick labels
    
    # fig.suptitle(f"{name[:5].capitalize()} {name[-1]} Assumptions", fontsize=20)
    # plt.tight_layout(rect=[0, 0, 1, 1.05]) 
    plt.tight_layout()

    if save_plots:
        basename = f'{model_type}_{name}'
        filepath = get_next_filename(basename, directory=f'./assumptions-plots/{name}/')
        plt.savefig(filepath)
    plt.show()
    plt.close()


    # PRINT RESULTS
    print(f'\nAssumption of normality fulfilled according to skewness and curtosis: {approx_normal}')
    print(f'Assumption of approximate normality fulfilled according to Shapiro-Wilk: {shapiro_normal}')
    print(f'Assumption of homogeneity fulfilled: {"✅" if p_levene > 0.05 else "❌"}')
    print(f'Assumption of absence of multicollinearity: {"❌" if (high_corr_spearman or high_corr_pearson) else "✅"}')


    # return approx_normal, pvalue > 0.05, p_levene > 0.05 


import os

def get_next_filename(base_name, extension=".png", directory="./assumptions-plots/"):
    i = 1
    while True:
        filename = f"{base_name}-{i}{extension}"
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            return filepath
        i += 1



from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm

def perform_anova(df, formula, random_effect=None, use_random=False):
    """
    df: DataFrame
    dependent: str, name of the dependent variable
    covariates: list of str, names of covariates (can be categorical or numeric)
    random_effect: str, optional column for random effect (e.g., user_ID)
    use_random: bool, whether to use mixed-effects model
    reference_groups: dict, optional, maps covariate name to its reference category (for categorical variables)
    """
    # Open a file and redirect stdout to it
    # sys.stdout = open('output.txt', 'w')

    if use_random and random_effect:
        print(f"--- Mixed Effects ANOVA ---\nFormula: {formula}")
        model = sm.MixedLM.from_formula(formula, groups=df[random_effect], data=df).fit()
        print(model.summary())
        df["fitted_values"] = model.fittedvalues
        print(df.groupby("attributedUserExplanationType")["fitted_values"].mean())

        icc = model.cov_re / (model.cov_re + model.scale)
        print(f"\nEstimated ICC from mixed model: {icc.values[0][0]:.4f}")


        
    else:
        print(f"--- Standard ANOVA ---\nFormula: {formula}")
        model = ols(formula, data=df).fit()
        anova_table = anova_lm(model, typ=1)
        print(model.summary())
        print(anova_table)

        df['residuals'] = model.resid

        # import seaborn as sns
        # import matplotlib.pyplot as plt

        # # plot residuals by participant to see variance --> if it changes a lot, then random effect for participant would be good
        # sns.boxplot(x='userID', y='residuals', data=df)
        # plt.xticks(rotation=90)
        # plt.title("Residuals by Participant")
        # plt.show()

        # # Mean residual per participant
        # grouped = df.groupby('userID')['residuals']
        # mean_per_participant = grouped.transform('mean')
        # grand_mean = df['residuals'].mean()

        # # 3. Between-subject variance (SSB)
        # participant_means = df.groupby('userID')['residuals'].mean()
        # n_per_participant = df.groupby('userID').size()
        # SSB = np.sum(n_per_participant * (participant_means - grand_mean) ** 2)

        # # 4. Within-subject variance (SSW)
        # SSW = np.sum((df['residuals'] - mean_per_participant) ** 2)

        # # 5. ICC
        # icc = SSB / (SSB + SSW)
        # print(f"Estimated ICC: {icc:.4f}")

    return model

def perform_pairwise_comparison(df, dependent, independent):
    # from statsmodels.stats.multicomp import pairwise_tukeyhsd

    print(f"--- Tukey Pairwise Comparison ---\n")

    # Tukey's HSD test
    tukey = pairwise_tukeyhsd(endog=df[dependent],      # dependent variable
                            groups=df[independent],  # group variable
                            alpha=0.1)              # significance level
    
    print(tukey)


def perform_anova_ordered(df, formula, order_map = None,  random_effect=None, use_random=False):
    if order_map is None: 
        order_map = {
            "none": 1,
            "featureImportance": 2,
            "point": 3,
            "interval": 4
            }

    df['ExplanationType_order'] = df['attributedUserExplanationType'].map(order_map)
    formula_ordered = formula.replace('C(attributedUserExplanationType, Treatment(reference="none"))', 'ExplanationType_order')

    if use_random and random_effect:
        print(f"--- Mixed Effects Trend Analysis ANOVA ---\nFormula: {formula}")
        model = sm.MixedLM.from_formula(formula_ordered, groups=df[random_effect], data=df).fit()
        print(model.summary())
    else:
        print(f"--- Standard ANOVA for Trend Analysis ---\nFormula: {formula_ordered}")
        model = ols(formula_ordered, data=df).fit()
        anova = anova_lm(model, typ=1)
        print(model.summary())
        print(anova)


    return model


def non_parametric_test(df, dependent, independent):
    print("\n--- Non-Parametric Test: Kruskal-Wallis H-test ---")
    groups = [group[dependent].values for name, group in df.groupby(independent)]
    stat, p = stats.kruskal(*groups)
    print(f"Kruskal-Wallis H-test: H={stat:.3f}, p={p:.3f}")
    return stat, p
    

# def show_high_correlations(df, threshold=0.8):
#     """
#     Prints pairs of features with absolute correlation above a specified threshold.
    
#     Parameters:
#         df (pd.DataFrame): DataFrame with numerical features.
#         threshold (float): Minimum absolute correlation to consider high.
#     """
#     df_numeric = df.select_dtypes(include=['number'])
#     corr_matrix = df_numeric.corr().abs()

#     # Select upper triangle of correlation matrix
#     upper = corr_matrix.where(
#         np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
#     )

#     # Find features with correlation above threshold
#     high_corr = [(col1, col2, corr_matrix.loc[col1, col2]) 
#                  for col1 in upper.columns 
#                  for col2 in upper.index 
#                  if pd.notnull(upper.loc[col2, col1]) and upper.loc[col2, col1] > threshold]

#     return high_corr, upper


import pandas as pd
import numpy as np

def show_high_correlations(df, threshold=0.8):
    """
    Computes Pearson correlation for numerical features and Spearman correlation
    for all features (numerical + categorical converted to codes).
    
    Parameters:
        df (pd.DataFrame): The input dataframe.
        threshold (float): Correlation threshold for flagging high correlations.
    
    Returns:
        - high_corr_pearson: List of (var1, var2, correlation) from Pearson
        - high_corr_spearman: List of (var1, var2, correlation) from Spearman
        - pearson_matrix: Pearson correlation matrix (numeric only)
        - spearman_matrix: Spearman correlation matrix (all features as numeric)
    """
    
    # --- Pearson: only numerical columns
    df_numeric = df.select_dtypes(include=['number'])
    pearson_matrix = df_numeric.corr(method='pearson').abs()

    upper_pearson = pearson_matrix.where(
        np.triu(np.ones(pearson_matrix.shape), k=1).astype(bool)
    )
    high_corr_pearson = [
        (col1, col2, upper_pearson.loc[col2, col1])
        for col1 in upper_pearson.columns
        for col2 in upper_pearson.index
        if pd.notnull(upper_pearson.loc[col2, col1]) and upper_pearson.loc[col2, col1] > threshold
    ]
    
    # --- Spearman: convert all to numeric
    df_rankable = df.copy()
    for col in df_rankable.select_dtypes(include=['category', 'object']).columns:
        df_rankable[col] = df_rankable[col].astype('category').cat.codes

    spearman_matrix = df_rankable.corr(method='spearman').abs()
    upper_spearman = spearman_matrix.where(
        np.triu(np.ones(spearman_matrix.shape), k=1).astype(bool)
    )
    high_corr_spearman = [
        (col1, col2, upper_spearman.loc[col2, col1])
        for col1 in upper_spearman.columns
        for col2 in upper_spearman.index
        if pd.notnull(upper_spearman.loc[col2, col1]) and upper_spearman.loc[col2, col1] > threshold
    ]

    return high_corr_pearson, high_corr_spearman, pearson_matrix, spearman_matrix



def main(hypothesis, phase, dependent, data_path ='./data/', models_to_train=None, reference_groups={}, transform = None, 
         use_scaled_data = False, save_plots=False, aggregated=False):
    
    # Open a file and redirect stdout to it
    sys.stdout = open('output.txt', 'w')

    if use_scaled_data:
        suffix = '_scaled'
    else:
        suffix = ''

    # load the data
    data, houses, demographics, batch_info = load_valid_data(data_path)

    if hypothesis=='h1':
        print('##########################\n')
        print('HYPOTHESIS 1: There is an effect of explanation type on performance.')

        print(f'''Testing {hypothesis} with data from {phase}. \nThe dependent variable is the gap in house price between users' prediction and the true price.
              ''')
        print('##########################\n')
        
        phase_data = clean_data(data, demographics, houses, batch_info, phase=phase, transformation = transform)

        print('\n###############################################################################################\n')

        # if phase=='p0':
            # dependent = 'log_data'
        if transform is None:
            dependent = dependent + suffix
        else:
            dependent = f'{transform}_{dependent}{suffix}'
        
        if aggregated:
            # group by user and calculate mse
            phase_data['squared_error'] = phase_data['boxcox_gap_house_price_scaled'] ** 2

            # get average MSE per participant
            # mse_per_participant = phase_data.groupby('userID')['squared_error'].mean().reset_index()
            # get RMSE per participant
            rmse_per_participant = phase_data.groupby('userID')['squared_error'].mean().apply(np.sqrt).reset_index()


            # rename the column
            rmse_per_participant.rename(columns={'squared_error': 'average_rmse'}, inplace=True)
            phase_data = pd.merge(rmse_per_participant, demographics, on='userID')

            dependent = 'average_rmse'

        get_descriptives(phase_data)

        print(phase_data.groupby("attributedUserExplanationType")[dependent].mean())

        print('\n------------------------------------------------------------------------------------------\n')
        perform_pairwise_comparison(phase_data, dependent, 'attributedUserExplanationType')

        # define models to train
        if models_to_train is None: 
            models_to_train = {'model1': ['attributedUserExplanationType']}

            reference_groups = {
                'attributedUserExplanationType': "none"
            }
        
        for name, covariates in models_to_train.items():
            terms = []
            for cov in covariates:
                if '*' in cov or ':' in cov:
                    # Interaction term, add directly
                    terms.append(cov)
                elif reference_groups and cov in reference_groups:
                    ref = reference_groups[cov]
                    terms.append(f'C({cov}, Treatment(reference="{ref}"))')
                else:
                    if phase_data[cov].dtype == 'object' or phase_data[cov].dtype.name == 'category':
                        terms.append(f'C({cov})')
                    else:
                        terms.append(cov)

            formula = f"{dependent} ~ " + " + ".join(terms)
            print(f"{name}: {formula}")


            print(f"\n--- Running {name} ---")
            print(f'Predictors: {covariates}')
            raw_covs = [c for c in covariates if '*' not in c and ':' not in c]
            
            ##### Standard ANOVA ####
            print('\n------------------------------------------------------------------------------------------\n')
            # try:
            model = perform_anova(phase_data, formula)
            check_anova_assumptions(phase_data, dependent, 'attributedUserExplanationType', raw_covs, model, 'standard', name, save_plots)
            # except: print('...model could not be fitted...')
            
            if not aggregated:
                ##### Mixed Effects ANOVA ####
                print('\n------------------------------------------------------------------------------------------\n')
                try: 
                    model = perform_anova(phase_data, formula, random_effect='userID', use_random=True)
                    check_anova_assumptions(phase_data, dependent, 'attributedUserExplanationType', raw_covs, model, 'mixed-effects', name, save_plots)
                except: print('...model could not be fitted...')

            print('\n###############################################################################################\n')


