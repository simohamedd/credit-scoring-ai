package com.simo.credit;

import java.util.List;

/** Trains the credit model and assesses example applicants. Run with: mvn -q compile exec:java */
public class CreditApp {

    public static void main(String[] args) {
        List<Applicant> all = SyntheticApplicants.generate(12_000, 11);
        List<Applicant> train = all.subList(0, 9_000);
        List<Applicant> test = all.subList(9_000, 12_000);

        long repaid = test.stream().filter(Applicant::repaid).count();
        double baseline = Math.max(repaid, test.size() - repaid) / (double) test.size();

        CrossValidation.Result cv = CrossValidation.chooseDepth(train, 5, 1, 7);
        System.out.println("Choosing tree depth with 5-fold cross-validation:");
        cv.meanAccuracyByDepth().forEach((d, acc) -> System.out.printf("  depth %d  %.1f%%%s%n", d, 100 * acc, d == cv.bestDepth() ? "  <- chosen" : ""));
        System.out.println();

        CreditScorer scorer = CreditScorer.train(train, cv.bestDepth());
        System.out.printf("Trained on %,d past loans, tested on %,d unseen applicants%n", train.size(), test.size());
        System.out.printf("Accuracy %.1f%%  (always guessing the most common outcome: %.1f%%)%n%n",
                100 * scorer.accuracy(test), 100 * baseline);
        System.out.printf("The model learned %d rules. The top three levels:%n", scorer.tree().leaves());
        System.out.println(scorer.tree().describe(3));

        assess(scorer, "Steady earner", new Applicant(52_000, 0.18, 9, 0, 8_000, false));
        assess(scorer, "Stretched borrower", new Applicant(24_000, 0.55, 2, 3, 15_000, false));
    }

    private static void assess(CreditScorer scorer, String label, Applicant a) {
        CreditScorer.Result r = scorer.assess(a);
        System.out.printf("%s: %s (repay chance %.0f%%)%n", label, r.outcome(), 100 * r.repayChance());
        r.reasons().forEach(reason -> System.out.println("   because " + reason));
        System.out.println();
    }
}
