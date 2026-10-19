package com.simo.credit;

/**
 * A loan applicant.
 *
 * @param annualIncome       gross yearly income in pounds
 * @param debtToIncome       monthly debt payments / monthly income, 0 to 1
 * @param creditHistoryYears how long they have had credit
 * @param missedPayments     missed payments in the last two years
 * @param loanAmount         amount requested in pounds
 * @param repaid             known outcome: did a past applicant like this repay? (training label)
 */
public record Applicant(double annualIncome, double debtToIncome, double creditHistoryYears,
                        int missedPayments, double loanAmount, boolean repaid) {

    public static final String[] FEATURES = {
        "annual income", "debt-to-income", "credit history (years)", "missed payments", "loan / income"
    };

    public double[] features() {
        return new double[] {annualIncome, debtToIncome, creditHistoryYears, missedPayments, loanAmount / annualIncome};
    }
}
