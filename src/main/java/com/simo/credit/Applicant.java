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

    /** Rejects applications that cannot describe a real person, so they are never scored by mistake. */
    public Applicant {
        if (!(annualIncome > 0) || Double.isInfinite(annualIncome))
            throw new IllegalArgumentException("Annual income must be a positive amount, got " + annualIncome);
        if (!(debtToIncome >= 0 && debtToIncome <= 1))
            throw new IllegalArgumentException("Debt-to-income must be between 0 and 1, got " + debtToIncome);
        if (!(creditHistoryYears >= 0) || Double.isInfinite(creditHistoryYears))
            throw new IllegalArgumentException("Credit history must be zero or more years, got " + creditHistoryYears);
        if (missedPayments < 0)
            throw new IllegalArgumentException("Missed payments cannot be negative, got " + missedPayments);
        if (!(loanAmount >= 0) || Double.isInfinite(loanAmount))
            throw new IllegalArgumentException("Loan amount cannot be negative, got " + loanAmount);
    }

    public double[] features() {
        return new double[] {annualIncome, debtToIncome, creditHistoryYears, missedPayments, loanAmount / annualIncome};
    }
}
