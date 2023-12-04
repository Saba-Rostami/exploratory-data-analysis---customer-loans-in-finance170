
# Exploratory Data Analysis - Customer Loans in Finance170

## Table of Contents
- [Description](#description)
- [Project Goals](#project-goals)
- [What I Learned](#what-i-learned)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [Feedback](#feedback)
- [License](#license)

## Description
The "Exploratory Data Analysis - Customer Loans in Finance170" project involves the analysis of a dataset containing information about customer loans. The aim is to explore and gain insights into the characteristics of customer loans.

## Project Goals
- Explore and visualize loan amounts.
- Analyze the impact of interest rates on loan performance.
- Investigate the relationship between employment length and loan status.
- Identify trends in loan purposes and default rates.

## What I Learned
Experience in:
- Cleaning and preprocessing raw data.
- Utilizing descriptive statistics and visualizations.
- Drawing insights from data.

## Installation
1. Clone the repository.
   ```bash
   git clone https://github.com/your-username/exploratory-data-analysis---customer-loans-in-finance170.git
   cd exploratory-data-analysis---customer-loans-in-finance170

## Usage

Insights into customer loans and EDA reference.

## Project Structure

   data: Contains the dataset.
   notebooks: Jupyter notebooks.
   loan_data_analysis.ipynb: Main notebook.
   visualizations: Output directory for visualizations.

## Dataset

The dataset used for this analysis is sourced from Finance170 and includes various features related to customer loans.

### Columns:
   * id: Unique id of the loan
   * member_id: Id of the member to took out the loan
   * loan_amount: Amount of loan the applicant received
   * funded_amount: The total amount committed to the loan at that point in time
   * funded_amount_inv: The total amount committed by investors for that loan at that point in time
   * term: The number of monthly payments for the loan
   * int_rate (APR): Annual (APR) interest rate of the loan
   * instalment: The monthly payment owned by the borrower. This is inclusive of the interest.
   * grade: Loan company (LC) assigned loan grade
   * sub_grade: LC assigned loan sub grade
   * employment_length: Employment length in years
   * home_ownership: The home ownership status provided by the borrower
   * annual_inc: The annual income of the borrower
   * verification_status: Indicates whether the borrowers income was verified by the LC or the income source was verified
   * issue_date: Issue date of the loan
   * loan_status: Current status of the loan
   * payment_plan: Indicates if a payment plan is in place for the loan. Indication borrower is struggling to pay.
   * purpose: A category provided by the borrower for the loan request
   * dti: A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income
   * delinq_2yr: The number of 30+ days past-due payments in the borrower's credit file for the past 2 years
   * earliest_credit_line: The month the borrower's earliest reported credit line was opened
   * inq_last_6mths: The number of inquiries in past 6 months (excluding auto and mortgage inquiries)
   * mths_since_last_record: The number of months' since the last public record
   * open_accounts: The number of open credit lines in the borrower's credit file
   total_accounts: The total number of credit lines currently in the borrower's credit file
   * out_prncp: Remaining outstanding principal for total amount funded
   * out_prncp_inv: Remaining outstanding principal for portion of total amount funded by investors
   * total_payment`: Payments received to date for total amount funded
   * total_rec_int: Interest received to date
   * total_rec_late_fee: Late fees received to date
   * recoveries: Post charge off gross recovery
   * collection_recovery_fee: Post charge off collection fee
   * last_payment_date: Date on which last month payment was received
   * last_payment_amount: Last total payment amount received
   * next_payment_date: Next scheduled payment date
   * last_credit_pull_date: The most recent month LC pulled credit for this loan
   * collections_12_mths_ex_med: Number of collections in 12 months' excluding medical collections
   * mths_since_last_major_derog: Months' since most recent 90-day or worse rating
   * policy_code: Publicly available policy_code=1 new products not publicly available policy_code=2
   * application_type: Indicates whether the loan is an individual application or a joint application with two co-borrowers

## Contributing

Feel free to contribute, provide feedback, or suggest improvements. 

## Feedback

For feedback or questions, open an issue.

## License

This project is licensed under the MIT License.
