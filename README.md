# Project Name
> The project entails analyzing loan data to identify patterns predicting loan default, crucial for making informed loan approval decisions. The EDA approach helps in understanding how various consumer and loan attributes correlate with the likelihood of default. The aim is to minimize credit losses by identifying risky loan applicants, using insights to adjust lending strategies and enhance risk assessment. This case study serves as an application of EDA techniques in real-world financial scenarios, offering insights into the domain of risk analytics.


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Recommendations](#recommendations)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
* Background
This project is an Exploratory Data Analysis (EDA) case study in the banking and financial services sector, focusing on risk analytics. The primary goal is to apply EDA techniques to understand the driver variables for behind loan default, and mitigate the risks associated with lending loans.

* Business Problem
The project addresses a key challenge for a consumer finance company specializing in various types of loans. The core issue is managing the risk of credit loss, which occurs when borrowers default on their loans. The project aims to identify high-risk loan applicants to make informed decisions on loan approvals, potentially reducing financial losses.

* Dataset Used
The analysis utilizes a dataset containing historical loan application data. It includes detailed information about past loan applicants, such as loan amount, term, interest rate, employment length, home ownership status, annual income, and whether they defaulted (charged-off) or not. The initial dataset comprised 39,717 rows and 111 columns, which was cleaned and preprocessed to 36,789 rows and 28 columns. The cleaned dataset was then used for various analytical techniques to identify patterns indicating the likelihood of loan default.


<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions
- Key Predictive Variables: 
    - Several features emerged as strong predictors after binning and transformation, enhancing predictive accuracy.
        - Loan Term, Grade, Zip Code, State, and Public Record Bankruptcies independently show strong predictive power.
        - Loan Amount, Interest Rate, Installment, Annual Income, DTI, Inquiries in Last 6 Months, Public Records, and Revolving Utilization are particularly strong after categorization.
- Binning Effectiveness:
    - Binned categories in various features like loan amount, interest rate, installment, annual income, and revolving utilization significantly improve their predictive strength.
- Loan amounts categorized as 'Medium-High' and 'Very High' exhibit a significant increase in default risk.
- A clear borrower preference for 36-month loan terms was observed, with 60-month terms showing higher default risks.
- Interest rates above 12% are associated with a higher likelihood of default, particularly notable for rates exceeding 14.4%.
- The frequency of 'Charged Off' loans increases with higher loan installments, especially beyond $820.
- Loans with 'F' and 'G' grades tend to default across various DTI ratios, while 'A' and 'B' grades show resilience even at high DTI levels.
- Employment length alone is not a conclusive predictor of loan repayment, demonstrating similar default rates across varied employment durations.
- Homeownership patterns, particularly with 'Others' indicating higher default risks, moderately influence loan status.
- Annual income categories reveal higher default risks in lower income brackets, particularly below $40,000.
- The presence of derogatory public records and bankruptcies significantly increases the likelihood of loan charge-offs.
- Geographic Location Analysis: The study reveals significant variations in loan defaults across different states and zip codes. California, New York, Florida, and Texas have the highest number of loans, with a notable concentration in certain zip codes. Notably, loans from Florida, Missouri, and California exhibit higher default rates, indicating geographic factors play a crucial role in predicting loan defaults. This insight is key for tailoring risk management strategies to specific regions.
- Strong predictive insights were gleaned from analyzing loan term, grade, zip code, state, and public record bankruptcies, with data transformations enhancing predictive accuracy.

## Recommendations
- Implement Risk-Based Pricing: 
    - Adjust interest rates based on loan amount categories, especially for higher-risk 'Medium-High' and 'Very High' loans.
- Loan Term Strategy: 
    - Promote 36-month loans more aggressively due to their lower default rates compared to 60-month loans.
- Interest Rate Monitoring: 
    - Apply stricter scrutiny for loans with interest rates above 12%, and consider additional risk mitigation for rates above 14.4%.
- Installment Payment Analysis: 
    - Monitor loans with higher installment payments, particularly those exceeding $820, for potential default risks.
- Grade-Based Lending Policies: 
    - Exercise caution with loans categorized as 'F' and 'G' grades, given their higher propensity to default.
- Employment and Income Verification: 
    - Strengthen verification processes for employment length and annual income, especially for borrowers in lower income brackets.
- Geographic Risk Management: 
    - Develop region-specific lending strategies focusing on areas with higher default rates, such as Florida, Missouri, and California.
- Home Ownership Evaluation: 
    - Consider home ownership status as a moderate risk factor, especially for 'Others' category.
- Public Record Scrutiny: 
    - Pay special attention to borrowers with derogatory public records or bankruptcies, as they significantly increase default likelihood.
- Data-Driven Lending Decisions: 
    - Utilize the insights from the analysis, such as the importance of loan term, grade, zip code, state, and bankruptcy records, to make more informed lending decisions.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- Python - version 3.11.3
- pandas - version 1.5.3
- numpy - version 1.24.3
- matplotlib - 3.7.1
- seaborn - 0.12.2
- anaconda - 23.5.2

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Contact
Created by [@AnirbanG-git] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->